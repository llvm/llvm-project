//===--- MemberwiseConstructor.cpp - Generate C++ constructor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AST.h"
#include "ParsedAST.h"
#include "refactor/InsertionPoint.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {

// A tweak that adds a C++ constructor which initializes each member.
//
// Given:
//   struct S{ int x; unique_ptr<double> y; };
// the tweak inserts the constructor:
//   S(int x, unique_ptr<double> y) : x(x), y(std::move(y)) {}
//
// We place the constructor inline, other tweaks are available to outline it.
class MemberwiseConstructor : public Tweak {
public:
  const char *id() const override final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  std::string title() const override {
    return llvm::formatv("Define constructor");
  }

  bool prepare(const Selection &Inputs) override {
    // This tweak assumes move semantics.
    if (!Inputs.AST->getLangOpts().CPlusPlus11)
      return false;

    // Trigger only on class definitions.
    if (auto *N = Inputs.ASTSelection.commonAncestor())
      Class = N->ASTNode.get<CXXRecordDecl>();
    if (!Class || !Class->isThisDeclarationADefinition() || Class->isUnion() ||
        Class->getDeclName().isEmpty())
      return false;

    dlog("MemberwiseConstructor for {0}?", Class->getName());
    // For now, don't support nontrivial initialization of bases.
    for (const CXXBaseSpecifier &Base : Class->bases()) {
      const auto *BaseClass = Base.getType()->getAsCXXRecordDecl();
      if (!BaseClass || !BaseClass->hasDefaultConstructor()) {
        dlog("  can't construct base {0}", Base.getType().getAsString());
        return false;
      }
    }

    // We don't want to offer the tweak if there's a similar constructor.
    // For now, only offer it if all constructors are special members.
    for (const CXXConstructorDecl *CCD : Class->ctors()) {
      if (!CCD->isDefaultConstructor() && !CCD->isCopyOrMoveConstructor()) {
        dlog("  conflicting constructor");
        return false;
      }
    }

    // Examine the fields to see which ones we should initialize.
    for (const FieldDecl *D : Class->fields()) {
      switch (FieldAction A = considerField(D)) {
      case Fail:
        dlog("  difficult field {0}", D->getName());
        return false;
      case Skip:
        dlog("  (skipping field {0})", D->getName());
        break;
      default:
        Fields.push_back({D, A});
        break;
      }
    }
    // Only offer the tweak if we have some fields to initialize.
    if (Fields.empty()) {
      dlog("  no fields to initialize");
      return false;
    }

    return true;
  }

  Expected<Effect> apply(const Selection &Inputs) override {
    std::string Code = buildCode();
    // Prefer to place the new constructor...
    std::vector<Anchor> Anchors = {
        // Below special constructors.
        {[](const Decl *D) {
           if (const auto *CCD = llvm::dyn_cast<CXXConstructorDecl>(D))
             return CCD->isDefaultConstructor();
           return false;
         },
         Anchor::Below},
        // Above other constructors
        {[](const Decl *D) { return llvm::isa<CXXConstructorDecl>(D); },
         Anchor::Above},
        // At the top of the public section
        {[](const Decl *D) { return true; }, Anchor::Above},
    };
    auto Edit = insertDecl(Code, *Class, std::move(Anchors), AS_public);
    if (!Edit)
      return Edit.takeError();
    return Effect::mainFileEdit(Inputs.AST->getSourceManager(),
                                tooling::Replacements{std::move(*Edit)});
  }

private:
  enum FieldAction {
    Fail,    // Disallow the tweak, we can't handle this field.
    Skip,    // Do not initialize this field, but allow the tweak anyway.
    Move,    // Pass by value and std::move into place
    Copy,    // Pass by value and copy into place
    CopyRef, // Pass by const ref and copy into place
  };
  FieldAction considerField(const FieldDecl *Field) const {
    if (Field->hasInClassInitializer())
      return Skip;
    if (!Field->getIdentifier())
      return Fail;

    // Decide what to do based on the field type.
    class Visitor : public TypeVisitor<Visitor, FieldAction> {
    public:
      Visitor(const ASTContext &Ctx) : Ctx(Ctx) {}
      const ASTContext &Ctx;

      // If we don't understand the type, assume we can't handle it.
      FieldAction VisitType(const Type *T) { return Fail; }
      FieldAction VisitRecordType(const RecordType *T) {
        if (const auto *D = T->getAsCXXRecordDecl())
          return considerClassValue(*D);
        return Fail;
      }
      FieldAction VisitBuiltinType(const BuiltinType *T) {
        if (T->isInteger() || T->isFloatingPoint() || T->isNullPtrType())
          return Copy;
        return Fail;
      }
      FieldAction VisitObjCObjectPointerType(const ObjCObjectPointerType *) {
        return Ctx.getLangOpts().ObjCAutoRefCount ? Copy : Fail;
      }
      FieldAction VisitAttributedType(const AttributedType *T) {
        return Visit(T->getModifiedType().getCanonicalType().getTypePtr());
      }
#define ALWAYS(T, Action)                                                      \
  FieldAction Visit##T##Type(const T##Type *) { return Action; }
      // Trivially copyable types (pointers and numbers).
      ALWAYS(Pointer, Copy);
      ALWAYS(MemberPointer, Copy);
      ALWAYS(Reference, Copy);
      ALWAYS(Complex, Copy);
      ALWAYS(Enum, Copy);
      // These types are dependent (when canonical) and likely to be classes.
      // Move is a reasonable generic option.
      ALWAYS(DependentName, Move);
      ALWAYS(UnresolvedUsing, Move);
      ALWAYS(TemplateTypeParm, Move);
      ALWAYS(TemplateSpecialization, Move);
    };
#undef ALWAYS
    return Visitor(Class->getASTContext())
        .Visit(Field->getType().getCanonicalType().getTypePtr());
  }

  // Decide what to do with a field of type C.
  static FieldAction considerClassValue(const CXXRecordDecl &C) {
    if (!C.hasDefinition())
      return Skip;
    // We can't always tell if C is copyable/movable without doing Sema work.
    // We assume operations are possible unless we can prove not.
    bool CanCopy = C.hasUserDeclaredCopyConstructor() ||
                   C.needsOverloadResolutionForCopyConstructor() ||
                   !C.defaultedCopyConstructorIsDeleted();
    bool CanMove = C.hasUserDeclaredMoveConstructor() ||
                   (C.needsOverloadResolutionForMoveConstructor() ||
                    !C.defaultedMoveConstructorIsDeleted());
    bool CanDefaultConstruct = C.hasDefaultConstructor();
    if (C.hasUserDeclaredCopyConstructor() ||
        C.hasUserDeclaredMoveConstructor()) {
      for (const CXXConstructorDecl *CCD : C.ctors()) {
        bool IsUsable = !CCD->isDeleted() && CCD->getAccess() == AS_public;
        if (CCD->isCopyConstructor())
          CanCopy = CanCopy && IsUsable;
        if (CCD->isMoveConstructor())
          CanMove = CanMove && IsUsable;
        if (CCD->isDefaultConstructor())
          CanDefaultConstruct = IsUsable;
      }
    }
    dlog("  {0} CanCopy={1} CanMove={2} TriviallyCopyable={3}", C.getName(),
         CanCopy, CanMove, C.isTriviallyCopyable());
    if (CanCopy && C.isTriviallyCopyable())
      return Copy;
    if (CanMove)
      return Move;
    if (CanCopy)
      return CopyRef;
    // If it's neither copyable nor movable, then default construction is
    // likely to make sense (example: std::mutex).
    if (CanDefaultConstruct)
      return Skip;
    return Fail;
  }

  std::string buildCode() const {
    std::string S;
    llvm::raw_string_ostream OS(S);

    if (Fields.size() == 1)
      OS << "explicit ";
    OS << Class->getName() << "(";
    const char *Sep = "";
    for (const FieldInfo &Info : Fields) {
      OS << Sep;
      QualType ParamType = Info.Field->getType().getLocalUnqualifiedType();
      if (Info.Action == CopyRef)
        ParamType = Class->getASTContext().getLValueReferenceType(
            ParamType.withConst());
      OS << printType(ParamType, *Class,
                      /*Placeholder=*/paramName(Info.Field));
      Sep = ", ";
    }
    OS << ")";
    Sep = " : ";
    for (const FieldInfo &Info : Fields) {
      OS << Sep << Info.Field->getName() << "(";
      if (Info.Action == Move)
        OS << "std::move("; // FIXME: #include <utility> too
      OS << paramName(Info.Field);
      if (Info.Action == Move)
        OS << ")";
      OS << ")";
      Sep = ", ";
    }
    OS << " {}\n";

    return S;
  }

  llvm::StringRef paramName(const FieldDecl *Field) const {
    return Field->getName().trim("_");
  }

  const CXXRecordDecl *Class = nullptr;
  struct FieldInfo {
    const FieldDecl *Field;
    FieldAction Action;
  };
  std::vector<FieldInfo> Fields;
};
REGISTER_TWEAK(MemberwiseConstructor)

} // namespace
} // namespace clangd
} // namespace clang
