//===--- InlayHints.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InlayHints.h"
#include "../clang-tidy/utils/DesignatedInitializers.h"
#include "AST.h"
#include "Config.h"
#include "HeuristicResolver.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace clang {
namespace clangd {
namespace {

// For now, inlay hints are always anchored at the left or right of their range.
enum class HintSide { Left, Right };

void stripLeadingUnderscores(StringRef &Name) { Name = Name.ltrim('_'); }

// getDeclForType() returns the decl responsible for Type's spelling.
// This is the inverse of ASTContext::getTypeDeclType().
template <typename Ty, typename = decltype(((Ty *)nullptr)->getDecl())>
const NamedDecl *getDeclForTypeImpl(const Ty *T) {
  return T->getDecl();
}
const NamedDecl *getDeclForTypeImpl(const void *T) { return nullptr; }
const NamedDecl *getDeclForType(const Type *T) {
  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(TY, BASE)
#define TYPE(TY, BASE)                                                         \
  case Type::TY:                                                               \
    return getDeclForTypeImpl(llvm::cast<TY##Type>(T));
#include "clang/AST/TypeNodes.inc"
  }
  llvm_unreachable("Unknown TypeClass enum");
}

// getSimpleName() returns the plain identifier for an entity, if any.
llvm::StringRef getSimpleName(const DeclarationName &DN) {
  if (IdentifierInfo *Ident = DN.getAsIdentifierInfo())
    return Ident->getName();
  return "";
}
llvm::StringRef getSimpleName(const NamedDecl &D) {
  return getSimpleName(D.getDeclName());
}
llvm::StringRef getSimpleName(QualType T) {
  if (const auto *ET = llvm::dyn_cast<ElaboratedType>(T))
    return getSimpleName(ET->getNamedType());
  if (const auto *BT = llvm::dyn_cast<BuiltinType>(T)) {
    PrintingPolicy PP(LangOptions{});
    PP.adjustForCPlusPlus();
    return BT->getName(PP);
  }
  if (const auto *D = getDeclForType(T.getTypePtr()))
    return getSimpleName(D->getDeclName());
  return "";
}

// Returns a very abbreviated form of an expression, or "" if it's too complex.
// For example: `foo->bar()` would produce "bar".
// This is used to summarize e.g. the condition of a while loop.
std::string summarizeExpr(const Expr *E) {
  struct Namer : ConstStmtVisitor<Namer, std::string> {
    std::string Visit(const Expr *E) {
      if (E == nullptr)
        return "";
      return ConstStmtVisitor::Visit(E->IgnoreImplicit());
    }

    // Any sort of decl reference, we just use the unqualified name.
    std::string VisitMemberExpr(const MemberExpr *E) {
      return getSimpleName(*E->getMemberDecl()).str();
    }
    std::string VisitDeclRefExpr(const DeclRefExpr *E) {
      return getSimpleName(*E->getFoundDecl()).str();
    }
    std::string VisitCallExpr(const CallExpr *E) {
      return Visit(E->getCallee());
    }
    std::string
    VisitCXXDependentScopeMemberExpr(const CXXDependentScopeMemberExpr *E) {
      return getSimpleName(E->getMember()).str();
    }
    std::string
    VisitDependentScopeDeclRefExpr(const DependentScopeDeclRefExpr *E) {
      return getSimpleName(E->getDeclName()).str();
    }
    std::string VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *E) {
      return getSimpleName(E->getType()).str();
    }
    std::string VisitCXXTemporaryObjectExpr(const CXXTemporaryObjectExpr *E) {
      return getSimpleName(E->getType()).str();
    }

    // Step through implicit nodes that clang doesn't classify as such.
    std::string VisitCXXMemberCallExpr(const CXXMemberCallExpr *E) {
      // Call to operator bool() inside if (X): dispatch to X.
      if (E->getNumArgs() == 0 && E->getMethodDecl() &&
          E->getMethodDecl()->getDeclName().getNameKind() ==
              DeclarationName::CXXConversionFunctionName &&
          E->getSourceRange() ==
              E->getImplicitObjectArgument()->getSourceRange())
        return Visit(E->getImplicitObjectArgument());
      return ConstStmtVisitor::VisitCXXMemberCallExpr(E);
    }
    std::string VisitCXXConstructExpr(const CXXConstructExpr *E) {
      if (E->getNumArgs() == 1)
        return Visit(E->getArg(0));
      return "";
    }

    // Literals are just printed
    std::string VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
      return E->getValue() ? "true" : "false";
    }
    std::string VisitIntegerLiteral(const IntegerLiteral *E) {
      return llvm::to_string(E->getValue());
    }
    std::string VisitFloatingLiteral(const FloatingLiteral *E) {
      std::string Result;
      llvm::raw_string_ostream OS(Result);
      E->getValue().print(OS);
      // Printer adds newlines?!
      Result.resize(llvm::StringRef(Result).rtrim().size());
      return Result;
    }
    std::string VisitStringLiteral(const StringLiteral *E) {
      std::string Result = "\"";
      if (E->containsNonAscii()) {
        Result += "...";
      } else if (E->getLength() > 10) {
        Result += E->getString().take_front(7);
        Result += "...";
      } else {
        llvm::raw_string_ostream OS(Result);
        llvm::printEscapedString(E->getString(), OS);
      }
      Result.push_back('"');
      return Result;
    }

    // Simple operators. Motivating cases are `!x` and `I < Length`.
    std::string printUnary(llvm::StringRef Spelling, const Expr *Operand,
                           bool Prefix) {
      std::string Sub = Visit(Operand);
      if (Sub.empty())
        return "";
      if (Prefix)
        return (Spelling + Sub).str();
      Sub += Spelling;
      return Sub;
    }
    bool InsideBinary = false; // No recursing into binary expressions.
    std::string printBinary(llvm::StringRef Spelling, const Expr *LHSOp,
                            const Expr *RHSOp) {
      if (InsideBinary)
        return "";
      llvm::SaveAndRestore InBinary(InsideBinary, true);

      std::string LHS = Visit(LHSOp);
      std::string RHS = Visit(RHSOp);
      if (LHS.empty() && RHS.empty())
        return "";

      if (LHS.empty())
        LHS = "...";
      LHS.push_back(' ');
      LHS += Spelling;
      LHS.push_back(' ');
      if (RHS.empty())
        LHS += "...";
      else
        LHS += RHS;
      return LHS;
    }
    std::string VisitUnaryOperator(const UnaryOperator *E) {
      return printUnary(E->getOpcodeStr(E->getOpcode()), E->getSubExpr(),
                        !E->isPostfix());
    }
    std::string VisitBinaryOperator(const BinaryOperator *E) {
      return printBinary(E->getOpcodeStr(E->getOpcode()), E->getLHS(),
                         E->getRHS());
    }
    std::string VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *E) {
      const char *Spelling = getOperatorSpelling(E->getOperator());
      // Handle weird unary-that-look-like-binary postfix operators.
      if ((E->getOperator() == OO_PlusPlus ||
           E->getOperator() == OO_MinusMinus) &&
          E->getNumArgs() == 2)
        return printUnary(Spelling, E->getArg(0), false);
      if (E->isInfixBinaryOp())
        return printBinary(Spelling, E->getArg(0), E->getArg(1));
      if (E->getNumArgs() == 1) {
        switch (E->getOperator()) {
        case OO_Plus:
        case OO_Minus:
        case OO_Star:
        case OO_Amp:
        case OO_Tilde:
        case OO_Exclaim:
        case OO_PlusPlus:
        case OO_MinusMinus:
          return printUnary(Spelling, E->getArg(0), true);
        default:
          break;
        }
      }
      return "";
    }
  };
  return Namer{}.Visit(E);
}

// Determines if any intermediate type in desugaring QualType QT is of
// substituted template parameter type. Ignore pointer or reference wrappers.
bool isSugaredTemplateParameter(QualType QT) {
  static auto PeelWrapper = [](QualType QT) {
    // Neither `PointerType` nor `ReferenceType` is considered as sugared
    // type. Peel it.
    QualType Peeled = QT->getPointeeType();
    return Peeled.isNull() ? QT : Peeled;
  };

  // This is a bit tricky: we traverse the type structure and find whether or
  // not a type in the desugaring process is of SubstTemplateTypeParmType.
  // During the process, we may encounter pointer or reference types that are
  // not marked as sugared; therefore, the desugar function won't apply. To
  // move forward the traversal, we retrieve the pointees using
  // QualType::getPointeeType().
  //
  // However, getPointeeType could leap over our interests: The QT::getAs<T>()
  // invoked would implicitly desugar the type. Consequently, if the
  // SubstTemplateTypeParmType is encompassed within a TypedefType, we may lose
  // the chance to visit it.
  // For example, given a QT that represents `std::vector<int *>::value_type`:
  //  `-ElaboratedType 'value_type' sugar
  //    `-TypedefType 'vector<int *>::value_type' sugar
  //      |-Typedef 'value_type'
  //      `-SubstTemplateTypeParmType 'int *' sugar class depth 0 index 0 T
  //        |-ClassTemplateSpecialization 'vector'
  //        `-PointerType 'int *'
  //          `-BuiltinType 'int'
  // Applying `getPointeeType` to QT results in 'int', a child of our target
  // node SubstTemplateTypeParmType.
  //
  // As such, we always prefer the desugared over the pointee for next type
  // in the iteration. It could avoid the getPointeeType's implicit desugaring.
  while (true) {
    if (QT->getAs<SubstTemplateTypeParmType>())
      return true;
    QualType Desugared = QT->getLocallyUnqualifiedSingleStepDesugaredType();
    if (Desugared != QT)
      QT = Desugared;
    else if (auto Peeled = PeelWrapper(Desugared); Peeled != QT)
      QT = Peeled;
    else
      break;
  }
  return false;
}

// A simple wrapper for `clang::desugarForDiagnostic` that provides optional
// semantic.
std::optional<QualType> desugar(ASTContext &AST, QualType QT) {
  bool ShouldAKA = false;
  auto Desugared = clang::desugarForDiagnostic(AST, QT, ShouldAKA);
  if (!ShouldAKA)
    return std::nullopt;
  return Desugared;
}

// Apply a series of heuristic methods to determine whether or not a QualType QT
// is suitable for desugaring (e.g. getting the real name behind the using-alias
// name). If so, return the desugared type. Otherwise, return the unchanged
// parameter QT.
//
// This could be refined further. See
// https://github.com/clangd/clangd/issues/1298.
QualType maybeDesugar(ASTContext &AST, QualType QT) {
  // Prefer desugared type for name that aliases the template parameters.
  // This can prevent things like printing opaque `: type` when accessing std
  // containers.
  if (isSugaredTemplateParameter(QT))
    return desugar(AST, QT).value_or(QT);

  // Prefer desugared type for `decltype(expr)` specifiers.
  if (QT->isDecltypeType())
    return QT.getCanonicalType();
  if (const AutoType *AT = QT->getContainedAutoType())
    if (!AT->getDeducedType().isNull() &&
        AT->getDeducedType()->isDecltypeType())
      return QT.getCanonicalType();

  return QT;
}

// Given a callee expression `Fn`, if the call is through a function pointer,
// try to find the declaration of the corresponding function pointer type,
// so that we can recover argument names from it.
// FIXME: This function is mostly duplicated in SemaCodeComplete.cpp; unify.
static FunctionProtoTypeLoc getPrototypeLoc(Expr *Fn) {
  TypeLoc Target;
  Expr *NakedFn = Fn->IgnoreParenCasts();
  if (const auto *T = NakedFn->getType().getTypePtr()->getAs<TypedefType>()) {
    Target = T->getDecl()->getTypeSourceInfo()->getTypeLoc();
  } else if (const auto *DR = dyn_cast<DeclRefExpr>(NakedFn)) {
    const auto *D = DR->getDecl();
    if (const auto *const VD = dyn_cast<VarDecl>(D)) {
      Target = VD->getTypeSourceInfo()->getTypeLoc();
    }
  }

  if (!Target)
    return {};

  // Unwrap types that may be wrapping the function type
  while (true) {
    if (auto P = Target.getAs<PointerTypeLoc>()) {
      Target = P.getPointeeLoc();
      continue;
    }
    if (auto A = Target.getAs<AttributedTypeLoc>()) {
      Target = A.getModifiedLoc();
      continue;
    }
    if (auto P = Target.getAs<ParenTypeLoc>()) {
      Target = P.getInnerLoc();
      continue;
    }
    break;
  }

  if (auto F = Target.getAs<FunctionProtoTypeLoc>()) {
    return F;
  }

  return {};
}

ArrayRef<const ParmVarDecl *>
maybeDropCxxExplicitObjectParameters(ArrayRef<const ParmVarDecl *> Params) {
  if (!Params.empty() && Params.front()->isExplicitObjectParameter())
    Params = Params.drop_front(1);
  return Params;
}

class TypeHintBuilder : public TypeVisitor<TypeHintBuilder> {
  QualType CurrentType;
  NestedNameSpecifier *CurrentNestedNameSpecifier;
  ASTContext &Context;
  StringRef MainFilePath;
  const PrintingPolicy &PP;
  SourceManager &SM;
  std::vector<InlayHintLabelPart> LabelChunks;
  bool AppendTrailingSpaceBeforeRightQual = true;

  void addLabel(llvm::function_ref<void(llvm::raw_ostream &)> NamePrinter,
                SourceLocation Location = SourceLocation()) {
    std::string Label;
    llvm::raw_string_ostream OS(Label);
    NamePrinter(OS);
    if (!Location.isValid())
      return addLabel(std::move(Label));
    auto &Name = LabelChunks.emplace_back();
    Name.value = std::move(Label);
    Name.location = makeLocation(Context, Location, MainFilePath);
  }

  void addLabel(std::string Label) {
    if (LabelChunks.empty()) {
      LabelChunks.emplace_back(std::move(Label));
      return;
    }
    auto &Back = LabelChunks.back();
    if (Back.location) {
      LabelChunks.emplace_back(std::move(Label));
      return;
    }
    // Let's combine the "unclickable" pieces together.
    Back.value += std::move(Label);
  }

  void printTemplateArgumentList(llvm::ArrayRef<TemplateArgument> Args) {
    unsigned Size = Args.size();
    for (unsigned I = 0; I < Size; ++I) {
      auto &TA = Args[I];
      if (PP.SuppressDefaultTemplateArgs && TA.getIsDefaulted())
        continue;
      if (I)
        addLabel(", ");
      printTemplateArgument(TA);
    }
  }

  void printTemplateArgument(const TemplateArgument &TA) {
    switch (TA.getKind()) {
    case TemplateArgument::Pack:
      return printTemplateArgumentList(TA.pack_elements());
    case TemplateArgument::Type:
      return VisitQualType(TA.getAsType());
    // TODO: Add support for NTTP arguments.
    case TemplateArgument::Expression:
    case TemplateArgument::StructuralValue:
    case TemplateArgument::Null:
    case TemplateArgument::Declaration:
    case TemplateArgument::NullPtr:
    case TemplateArgument::Integral:
    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
      break;
    }
    std::string Label;
    llvm::raw_string_ostream OS(Label);
    TA.print(PP, OS, /*IncludeType=*/true);
    addLabel(std::move(Label));
  }

  void handleTemplateSpecialization(llvm::StringRef TemplateId,
                                    llvm::ArrayRef<TemplateArgument> Args,
                                    SourceLocation Location) {
    addLabel([&](llvm::raw_ostream &OS) { OS << TemplateId; }, Location);
    addLabel("<");
    printTemplateArgumentList(Args);
    addLabel(">");
  }

  void maybeAddQualifiers(bool AppendSpaceToQuals) {
    addLabel([&](llvm::raw_ostream &OS) {
      CurrentType.split().Quals.print(
          OS, PP, /*appendSpaceIfNonEmpty=*/AppendSpaceToQuals);
    });
  }

  // When printing a reference, the referenced type might also be a reference.
  // If so, we want to skip that before printing the inner type.
  static QualType skipTopLevelReferences(QualType T) {
    if (auto *Ref = T->getAs<ReferenceType>())
      return skipTopLevelReferences(Ref->getPointeeTypeAsWritten());
    return T;
  }

  static SourceLocation nameLocation(Decl *D, const SourceManager &SM) {
    // If this is a definition, find its *forward declaration* if possible.
    //
    // Per LSP specification, code actions, e.g., hover/go-to-def on the type
    // link, would be performed as if at the location we have given.
    //
    // Therefore, we should provide the type part with a location that points to
    // its declaration because we would otherwise take users to the
    // *declaration* if they're at the definition.
    if (auto *TD = dyn_cast<TemplateDecl>(D))
      D = TD->getTemplatedDecl();
    bool IsDefinition =
        isa<TagDecl>(D) && cast<TagDecl>(D)->isThisDeclarationADefinition();
    if (IsDefinition) {
      // Happy path: if the canonical declaration is a forward declaration.
      if (!cast<TagDecl>(D)->getCanonicalDecl()->isThisDeclarationADefinition())
        D = D->getCanonicalDecl();
      else {
        // Otherwise, look through the redeclarations.
        for (auto *Redecl : D->redecls())
          if (!cast<TagDecl>(Redecl)->isThisDeclarationADefinition()) {
            D = Redecl;
            break;
          }
      }
    }
    return ::clang::clangd::nameLocation(*D, SM);
  }

  // CanPrefixQualifiers - We prefer to print type qualifiers
  // before the type, so that we get "const int" instead of "int const", but we
  // can't do this if the type is complex.  For example if the type is "int*",
  // we *must* print "int * const", printing "const int *" is different.  Only
  // do this when the type expands to a simple string.
  // This is similar to the private function \p
  // TypePrinter::canPrefixQualifiers().
  // FIXME: Refactor and share the same implementation.
  static bool canPrefixQualifiers(const Type *T) {
    bool CanPrefixQualifiers = false;
    const Type *UnderlyingType = T;
    if (const auto *AT = dyn_cast<AutoType>(T))
      UnderlyingType = AT->desugar().getTypePtr();
    if (const auto *Subst = dyn_cast<SubstTemplateTypeParmType>(T))
      UnderlyingType = Subst->getReplacementType().getTypePtr();
    Type::TypeClass TC = UnderlyingType->getTypeClass();

    switch (TC) {
    case Type::Adjusted:
    case Type::Decayed:
    case Type::ArrayParameter:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::DependentAddressSpace:
    case Type::DependentVector:
    case Type::DependentSizedExtVector:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::DependentSizedMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Paren:
    case Type::PackExpansion:
    case Type::SubstTemplateTypeParm:
    case Type::MacroQualified:
    case Type::CountAttributed:
      CanPrefixQualifiers = false;
      break;
    default:
      CanPrefixQualifiers = true;
    }

    return CanPrefixQualifiers;
  }

  SourceLocation getPreferredLocationFromSpecialization(
      ClassTemplateSpecializationDecl *Specialization) const {
    SourceLocation Location;
    // Quirk as it is, the Specialization might have no associated forward
    // declarations. So we have to find them through the Pattern.
    if (!Specialization->isExplicitInstantiationOrSpecialization()) {
      auto Pattern = Specialization->getSpecializedTemplateOrPartial();
      if (auto *Template = Pattern.dyn_cast<ClassTemplateDecl *>())
        Location = nameLocation(Template, SM);
      if (auto *Template =
              Pattern.dyn_cast<ClassTemplatePartialSpecializationDecl *>())
        Location = nameLocation(Template, SM);
    } else
      Location = nameLocation(Specialization, SM);
    return Location;
  }

public:
  TypeHintBuilder(ASTContext &Context, StringRef MainFilePath,
                  const PrintingPolicy &PP, llvm::StringRef Prefix)
      : CurrentNestedNameSpecifier(nullptr), Context(Context),
        MainFilePath(MainFilePath), PP(PP), SM(Context.getSourceManager()) {
    LabelChunks.reserve(16);
    if (!Prefix.empty())
      addLabel(Prefix.str());
  }

  void VisitType(const Type *T) {
    // We should have handled qualifiers in VisitQualType(). Don't print them
    // twice.
    addLabel(QualType(T, /*Quals=*/0).getAsString());
  }

  void VisitQualType(QualType Q, bool AppendSpaceToTopLevelQuals = true,
                     NestedNameSpecifier *NNS = nullptr) {
    QualType PreviousType = CurrentType;
    NestedNameSpecifier *PreviousNNS = CurrentNestedNameSpecifier;

    CurrentType = Q;
    CurrentNestedNameSpecifier = NNS;
    bool CanPrefixQualifiers = canPrefixQualifiers(CurrentType.getTypePtr());
    bool PrevAppendTrailingSpaceBeforeRightQual =
        AppendTrailingSpaceBeforeRightQual;
    if (CanPrefixQualifiers)
      maybeAddQualifiers(/*AppendSpaceToQuals=*/true);

    TypeVisitor::Visit(Q.getTypePtr());

    bool HaveTrailingQuals =
        !CanPrefixQualifiers && !CurrentType.split().Quals.empty();
    if (AppendTrailingSpaceBeforeRightQual && HaveTrailingQuals)
      addLabel(" ");
    if (HaveTrailingQuals) {
      maybeAddQualifiers(/*AppendSpaceToQuals=*/AppendSpaceToTopLevelQuals);
      AppendTrailingSpaceBeforeRightQual =
          PrevAppendTrailingSpaceBeforeRightQual;
    }

    CurrentType = PreviousType;
    CurrentNestedNameSpecifier = PreviousNNS;
  }

  void VisitTagType(const TagType *TT) {
    auto *CXXRD = dyn_cast<CXXRecordDecl>(TT->getDecl());
    if (!CXXRD) {
      // This might be a C TagDecl.
      if (auto *RD = dyn_cast<RecordDecl>(TT->getDecl())) {
        // FIXME: Respect SuppressTagKeyword in other cases.
        if (!PP.SuppressTagKeyword && !RD->getTypedefNameForAnonDecl())
          addLabel(
              [&](llvm::raw_ostream &OS) { OS << RD->getKindName() << " "; });
        return addLabel(
            [&](llvm::raw_ostream &OS) { return RD->printName(OS, PP); },
            nameLocation(RD, SM));
      }
      return VisitType(TT);
    }
    // Note that we have cases where the type of a template specialization is
    // modeled as a RecordType rather than a TemplateSpecializationType. (Type
    // sugars are not preserved?)
    // Example:
    //
    // template <typename, typename = int>
    // struct A {};
    // A<float> bar[1];
    //
    // auto [value] = bar;
    //
    // The type of value is modeled as a RecordType here.

    // The ClassTemplateSpecializationDecl could be of TSK_Undeclared
    // kind. So handle the ClassTemplateSpecializationDecl case first for
    // template arguments.
    // E.g. when we're inside a range-based for loop:
    //   template <class T, class U> struct Pair;
    //   for (auto p : SmallVector<Pair<SourceLocation, SourceRange>>()) {}
    // (Pair is of TSK_Undeclared kind here.)
    // FIXME: Do we have other kinds of specializations?
    if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(CXXRD)) {
      std::string TemplateId;
      llvm::raw_string_ostream OS(TemplateId);
      CTSD->printName(OS);
      return handleTemplateSpecialization(
          TemplateId, CTSD->getTemplateArgs().asArray(),
          getPreferredLocationFromSpecialization(CTSD));
    }

    // We don't have a template arguments now. Find the name and its location.
    if (!CXXRD->getTemplateSpecializationKind())
      return addLabel(
          [&](llvm::raw_ostream &OS) { return CXXRD->printName(OS, PP); },
          nameLocation(CXXRD, SM));

    return VisitType(TT);
  }

  void VisitEnumType(const EnumType *ET) {
    return addLabel(
        [&](llvm::raw_ostream &OS) { return ET->getDecl()->printName(OS, PP); },
        nameLocation(ET->getDecl(), SM));
  }

  void VisitAutoType(const AutoType *AT) {
    if (!AT->isDeduced() || AT->getDeducedType()->isDecltypeType())
      return VisitType(AT);
    return VisitQualType(AT->getDeducedType());
  }

  void VisitElaboratedType(const ElaboratedType *ET) {
    if (auto *NNS = ET->getQualifier()) {
      switch (NNS->getKind()) {
      case NestedNameSpecifier::Identifier:
      case NestedNameSpecifier::Namespace:
      case NestedNameSpecifier::NamespaceAlias:
      case NestedNameSpecifier::Global:
      case NestedNameSpecifier::Super: {
        if (PP.SuppressScope)
          break;
        std::string Label;
        llvm::raw_string_ostream OS(Label);
        NNS->print(OS, PP);
        addLabel(std::move(Label));
        break;
      }
      case NestedNameSpecifier::TypeSpec:
      case NestedNameSpecifier::TypeSpecWithTemplate:
        if (PP.SuppressScope)
          break;
        // Do we need cv-qualifiers on type specifiers?
        VisitQualType(QualType(NNS->getAsType(), /*Quals=*/0));
        addLabel("::");
        break;
      }
    }
    return VisitQualType(ET->getNamedType(),
                         /*AppendSpaceToTopLevelQuals=*/true,
                         ET->getQualifier());
  }

  void VisitReferenceType(const ReferenceType *RT) {
    QualType Next = skipTopLevelReferences(RT->getPointeeTypeAsWritten());
    VisitQualType(Next);
    if (Next->getPointeeType().isNull())
      addLabel(" ");
    if (RT->isLValueReferenceType())
      addLabel("&");
    if (RT->isRValueReferenceType())
      addLabel("&&");
  }

  void VisitPointerType(const PointerType *PT) {
    // We don't want a trailing space after the last asterisk, if it is followed
    // by a qualifier. E.g. 'int *const' rather than 'int * const'.
    AppendTrailingSpaceBeforeRightQual = false;
    QualType Next = PT->getPointeeType();
    VisitQualType(Next);
    if (Next->getPointeeType().isNull())
      addLabel(" ");
    addLabel("*");
  }

  void VisitUsingType(const UsingType *UT) {
    addLabel([&](llvm::raw_ostream &OS) { UT->getFoundDecl()->printName(OS); },
             nameLocation(UT->getFoundDecl()->getIntroducer(), SM));
  }

  void VisitTypedefType(const TypedefType *TT) {
    addLabel([&](llvm::raw_ostream &OS) { TT->getDecl()->printName(OS); },
             nameLocation(TT->getDecl(), SM));
  }

  void VisitTemplateSpecializationType(const TemplateSpecializationType *TST) {
    SourceLocation Location;
    TemplateName Name = TST->getTemplateName();
    TemplateName::Qualified PrintQual = TemplateName::Qualified::AsWritten;
    switch (Name.getKind()) {
    case TemplateName::Template:
    case TemplateName::QualifiedTemplate: {
      if (QualifiedTemplateName *Qual = Name.getAsQualifiedTemplateName();
          Qual && Qual->getQualifier() == CurrentNestedNameSpecifier) {
        // We have handled the NNS in VisitElaboratedType(). Avoid printing it
        // twice.
        Name = Qual->getUnderlyingTemplate();
        PrintQual = TemplateName::Qualified::None;
      }
      [[fallthrough]];
    }
    case TemplateName::SubstTemplateTemplateParm:
    case TemplateName::UsingTemplate:
      Location = Name.getAsTemplateDecl()->getLocation();
      break;
    case TemplateName::OverloadedTemplate:
    case TemplateName::AssumedTemplate:
    case TemplateName::DependentTemplate:
    case TemplateName::SubstTemplateTemplateParmPack:
      // FIXME: Handle these cases.
      return VisitType(TST);
    }
    // Special case the ClassTemplateSpecializationDecl because
    // we want the location of an explicit specialization, if present.
    // FIXME: In practice, populating the location with that of the
    // specialization would still take us to the primary template because we're
    // actually sending a go-to-def request from the explicit specialization.
    if (auto *Specialization =
            dyn_cast_if_present<ClassTemplateSpecializationDecl>(
                TST->desugar().getCanonicalType()->getAsCXXRecordDecl()))
      Location = getPreferredLocationFromSpecialization(Specialization);
    std::string TemplateId;
    llvm::raw_string_ostream OS(TemplateId);
    Name.print(OS, PP, PrintQual);
    return handleTemplateSpecialization(TemplateId, TST->template_arguments(),
                                        Location);
  }

  void VisitDeducedTemplateSpecializationType(
      const DeducedTemplateSpecializationType *TST) {
    // FIXME: The TST->getTemplateName() might differ from the name of
    // DeducedType, e.g. when the deduction guide is formed against a type alias
    // Decl.
    return VisitQualType(TST->getDeducedType());
  }

  void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *ST) {
    return VisitQualType(ST->getReplacementType());
  }

  std::vector<InlayHintLabelPart> take() { return std::move(LabelChunks); }
};

unsigned lengthOfInlayHintLabel(llvm::ArrayRef<InlayHintLabelPart> Labels) {
  unsigned Size = 0;
  for (auto &P : Labels)
    Size += P.value.size();
  return Size;
}

struct Callee {
  // Only one of Decl or Loc is set.
  // Loc is for calls through function pointers.
  const FunctionDecl *Decl = nullptr;
  FunctionProtoTypeLoc Loc;
};

class InlayHintVisitor : public RecursiveASTVisitor<InlayHintVisitor> {
public:
  InlayHintVisitor(std::vector<InlayHint> &Results, ParsedAST &AST,
                   const Config &Cfg, std::optional<Range> RestrictRange)
      : Results(Results), AST(AST.getASTContext()), Tokens(AST.getTokens()),
        Cfg(Cfg), RestrictRange(std::move(RestrictRange)),
        MainFileID(AST.getSourceManager().getMainFileID()),
        MainFilePath(AST.tuPath()), Resolver(AST.getHeuristicResolver()),
        TypeHintPolicy(this->AST.getPrintingPolicy()) {
    bool Invalid = false;
    llvm::StringRef Buf =
        AST.getSourceManager().getBufferData(MainFileID, &Invalid);
    MainFileBuf = Invalid ? StringRef{} : Buf;

    TypeHintPolicy.SuppressScope = true; // keep type names short
    TypeHintPolicy.AnonymousTagLocations =
        false; // do not print lambda locations

    // Not setting PrintCanonicalTypes for "auto" allows
    // SuppressDefaultTemplateArgs (set by default) to have an effect.
  }

  bool VisitTypeLoc(TypeLoc TL) {
    if (const auto *DT = llvm::dyn_cast<DecltypeType>(TL.getType()))
      if (QualType UT = DT->getUnderlyingType(); !UT->isDependentType())
        addTypeHint(TL.getSourceRange(), UT, ": ");
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    // Weed out constructor calls that don't look like a function call with
    // an argument list, by checking the validity of getParenOrBraceRange().
    // Also weed out std::initializer_list constructors as there are no names
    // for the individual arguments.
    if (!E->getParenOrBraceRange().isValid() ||
        E->isStdInitListInitialization()) {
      return true;
    }

    Callee Callee;
    Callee.Decl = E->getConstructor();
    if (!Callee.Decl)
      return true;
    processCall(Callee, {E->getArgs(), E->getNumArgs()});
    return true;
  }

  // Carefully recurse into PseudoObjectExprs, which typically incorporate
  // a syntactic expression and several semantic expressions.
  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    Expr *SyntacticExpr = E->getSyntacticForm();
    if (isa<CallExpr>(SyntacticExpr))
      // Since the counterpart semantics usually get the identical source
      // locations as the syntactic one, visiting those would end up presenting
      // confusing hints e.g., __builtin_dump_struct.
      // Thus, only traverse the syntactic forms if this is written as a
      // CallExpr. This leaves the door open in case the arguments in the
      // syntactic form could possibly get parameter names.
      return RecursiveASTVisitor<InlayHintVisitor>::TraverseStmt(SyntacticExpr);
    // We don't want the hints for some of the MS property extensions.
    // e.g.
    // struct S {
    //   __declspec(property(get=GetX, put=PutX)) int x[];
    //   void PutX(int y);
    //   void Work(int y) { x = y; } // Bad: `x = y: y`.
    // };
    if (isa<BinaryOperator>(SyntacticExpr))
      return true;
    // FIXME: Handle other forms of a pseudo object expression.
    return RecursiveASTVisitor<InlayHintVisitor>::TraversePseudoObjectExpr(E);
  }

  bool VisitCallExpr(CallExpr *E) {
    if (!Cfg.InlayHints.Parameters)
      return true;

    bool IsFunctor = isFunctionObjectCallExpr(E);
    // Do not show parameter hints for user-defined literals or
    // operator calls except for operator(). (Among other reasons, the resulting
    // hints can look awkward, e.g. the expression can itself be a function
    // argument and then we'd get two hints side by side).
    if ((isa<CXXOperatorCallExpr>(E) && !IsFunctor) ||
        isa<UserDefinedLiteral>(E))
      return true;

    auto CalleeDecls = Resolver->resolveCalleeOfCallExpr(E);
    if (CalleeDecls.size() != 1)
      return true;

    Callee Callee;
    if (const auto *FD = dyn_cast<FunctionDecl>(CalleeDecls[0]))
      Callee.Decl = FD;
    else if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(CalleeDecls[0]))
      Callee.Decl = FTD->getTemplatedDecl();
    else if (FunctionProtoTypeLoc Loc = getPrototypeLoc(E->getCallee()))
      Callee.Loc = Loc;
    else
      return true;

    // N4868 [over.call.object]p3 says,
    // The argument list submitted to overload resolution consists of the
    // argument expressions present in the function call syntax preceded by the
    // implied object argument (E).
    //
    // As well as the provision from P0847R7 Deducing This [expr.call]p7:
    // ...If the function is an explicit object member function and there is an
    // implied object argument ([over.call.func]), the list of provided
    // arguments is preceded by the implied object argument for the purposes of
    // this correspondence...
    llvm::ArrayRef<const Expr *> Args = {E->getArgs(), E->getNumArgs()};
    // We don't have the implied object argument through a function pointer
    // either.
    if (const CXXMethodDecl *Method =
            dyn_cast_or_null<CXXMethodDecl>(Callee.Decl))
      if (IsFunctor || Method->hasCXXExplicitFunctionObjectParameter())
        Args = Args.drop_front(1);
    processCall(Callee, Args);
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *D) {
    if (auto *FPT =
            llvm::dyn_cast<FunctionProtoType>(D->getType().getTypePtr())) {
      if (!FPT->hasTrailingReturn()) {
        if (auto FTL = D->getFunctionTypeLoc())
          addReturnTypeHint(D, FTL.getRParenLoc());
      }
    }
    if (Cfg.InlayHints.BlockEnd && D->isThisDeclarationADefinition()) {
      // We use `printName` here to properly print name of ctor/dtor/operator
      // overload.
      if (const Stmt *Body = D->getBody())
        addBlockEndHint(Body->getSourceRange(), "", printName(AST, *D), "");
    }
    return true;
  }

  bool VisitForStmt(ForStmt *S) {
    if (Cfg.InlayHints.BlockEnd) {
      std::string Name;
      // Common case: for (int I = 0; I < N; I++). Use "I" as the name.
      if (auto *DS = llvm::dyn_cast_or_null<DeclStmt>(S->getInit());
          DS && DS->isSingleDecl())
        Name = getSimpleName(llvm::cast<NamedDecl>(*DS->getSingleDecl()));
      else
        Name = summarizeExpr(S->getCond());
      markBlockEnd(S->getBody(), "for", Name);
    }
    return true;
  }

  bool VisitCXXForRangeStmt(CXXForRangeStmt *S) {
    if (Cfg.InlayHints.BlockEnd)
      markBlockEnd(S->getBody(), "for", getSimpleName(*S->getLoopVariable()));
    return true;
  }

  bool VisitWhileStmt(WhileStmt *S) {
    if (Cfg.InlayHints.BlockEnd)
      markBlockEnd(S->getBody(), "while", summarizeExpr(S->getCond()));
    return true;
  }

  bool VisitSwitchStmt(SwitchStmt *S) {
    if (Cfg.InlayHints.BlockEnd)
      markBlockEnd(S->getBody(), "switch", summarizeExpr(S->getCond()));
    return true;
  }

  // If/else chains are tricky.
  //   if (cond1) {
  //   } else if (cond2) {
  //   } // mark as "cond1" or "cond2"?
  // For now, the answer is neither, just mark as "if".
  // The ElseIf is a different IfStmt that doesn't know about the outer one.
  llvm::DenseSet<const IfStmt *> ElseIfs; // not eligible for names
  bool VisitIfStmt(IfStmt *S) {
    if (Cfg.InlayHints.BlockEnd) {
      if (const auto *ElseIf = llvm::dyn_cast_or_null<IfStmt>(S->getElse()))
        ElseIfs.insert(ElseIf);
      // Don't use markBlockEnd: the relevant range is [then.begin, else.end].
      if (const auto *EndCS = llvm::dyn_cast<CompoundStmt>(
              S->getElse() ? S->getElse() : S->getThen())) {
        addBlockEndHint(
            {S->getThen()->getBeginLoc(), EndCS->getRBracLoc()}, "if",
            ElseIfs.contains(S) ? "" : summarizeExpr(S->getCond()), "");
      }
    }
    return true;
  }

  void markBlockEnd(const Stmt *Body, llvm::StringRef Label,
                    llvm::StringRef Name = "") {
    if (const auto *CS = llvm::dyn_cast_or_null<CompoundStmt>(Body))
      addBlockEndHint(CS->getSourceRange(), Label, Name, "");
  }

  bool VisitTagDecl(TagDecl *D) {
    if (Cfg.InlayHints.BlockEnd && D->isThisDeclarationADefinition()) {
      std::string DeclPrefix = D->getKindName().str();
      if (const auto *ED = dyn_cast<EnumDecl>(D)) {
        if (ED->isScoped())
          DeclPrefix += ED->isScopedUsingClassTag() ? " class" : " struct";
      };
      addBlockEndHint(D->getBraceRange(), DeclPrefix, getSimpleName(*D), ";");
    }
    return true;
  }

  bool VisitNamespaceDecl(NamespaceDecl *D) {
    if (Cfg.InlayHints.BlockEnd) {
      // For namespace, the range actually starts at the namespace keyword. But
      // it should be fine since it's usually very short.
      addBlockEndHint(D->getSourceRange(), "namespace", getSimpleName(*D), "");
    }
    return true;
  }

  bool VisitLambdaExpr(LambdaExpr *E) {
    FunctionDecl *D = E->getCallOperator();
    if (!E->hasExplicitResultType())
      addReturnTypeHint(D, E->hasExplicitParameters()
                               ? D->getFunctionTypeLoc().getRParenLoc()
                               : E->getIntroducerRange().getEnd());
    return true;
  }

  void addReturnTypeHint(FunctionDecl *D, SourceRange Range) {
    auto *AT = D->getReturnType()->getContainedAutoType();
    if (!AT || AT->getDeducedType().isNull())
      return;
    addTypeHint(Range, D->getReturnType(), /*Prefix=*/"-> ");
  }

  bool VisitVarDecl(VarDecl *D) {
    // Do not show hints for the aggregate in a structured binding,
    // but show hints for the individual bindings.
    if (auto *DD = dyn_cast<DecompositionDecl>(D)) {
      for (auto *Binding : DD->bindings()) {
        // For structured bindings, print canonical types. This is important
        // because for bindings that use the tuple_element protocol, the
        // non-canonical types would be "tuple_element<I, A>::type".
        if (auto Type = Binding->getType();
            !Type.isNull() && !Type->isDependentType())
          addTypeHint(Binding->getLocation(), Type.getCanonicalType(),
                      /*Prefix=*/": ");
      }
      return true;
    }

    if (auto *AT = D->getType()->getContainedAutoType()) {
      if (AT->isDeduced() && !D->getType()->isDependentType()) {
        // Our current approach is to place the hint on the variable
        // and accordingly print the full type
        // (e.g. for `const auto& x = 42`, print `const int&`).
        // Alternatively, we could place the hint on the `auto`
        // (and then just print the type deduced for the `auto`).
        addTypeHint(D->getLocation(), D->getType(), /*Prefix=*/": ");
      }
    }

    // Handle templates like `int foo(auto x)` with exactly one instantiation.
    if (auto *PVD = llvm::dyn_cast<ParmVarDecl>(D)) {
      if (D->getIdentifier() && PVD->getType()->isDependentType() &&
          !getContainedAutoParamType(D->getTypeSourceInfo()->getTypeLoc())
               .isNull()) {
        if (auto *IPVD = getOnlyParamInstantiation(PVD))
          addTypeHint(D->getLocation(), IPVD->getType(), /*Prefix=*/": ");
      }
    }

    return true;
  }

  ParmVarDecl *getOnlyParamInstantiation(ParmVarDecl *D) {
    auto *TemplateFunction = llvm::dyn_cast<FunctionDecl>(D->getDeclContext());
    if (!TemplateFunction)
      return nullptr;
    auto *InstantiatedFunction = llvm::dyn_cast_or_null<FunctionDecl>(
        getOnlyInstantiation(TemplateFunction));
    if (!InstantiatedFunction)
      return nullptr;

    unsigned ParamIdx = 0;
    for (auto *Param : TemplateFunction->parameters()) {
      // Can't reason about param indexes in the presence of preceding packs.
      // And if this param is a pack, it may expand to multiple params.
      if (Param->isParameterPack())
        return nullptr;
      if (Param == D)
        break;
      ++ParamIdx;
    }
    assert(ParamIdx < TemplateFunction->getNumParams() &&
           "Couldn't find param in list?");
    assert(ParamIdx < InstantiatedFunction->getNumParams() &&
           "Instantiated function has fewer (non-pack) parameters?");
    return InstantiatedFunction->getParamDecl(ParamIdx);
  }

  bool VisitInitListExpr(InitListExpr *Syn) {
    // We receive the syntactic form here (shouldVisitImplicitCode() is false).
    // This is the one we will ultimately attach designators to.
    // It may have subobject initializers inlined without braces. The *semantic*
    // form of the init-list has nested init-lists for these.
    // getUnwrittenDesignators will look at the semantic form to determine the
    // labels.
    assert(Syn->isSyntacticForm() && "RAV should not visit implicit code!");
    if (!Cfg.InlayHints.Designators)
      return true;
    if (Syn->isIdiomaticZeroInitializer(AST.getLangOpts()))
      return true;
    llvm::DenseMap<SourceLocation, std::string> Designators =
        tidy::utils::getUnwrittenDesignators(Syn);
    for (const Expr *Init : Syn->inits()) {
      if (llvm::isa<DesignatedInitExpr>(Init))
        continue;
      auto It = Designators.find(Init->getBeginLoc());
      if (It != Designators.end() &&
          !isPrecededByParamNameComment(Init, It->second))
        addDesignatorHint(Init->getSourceRange(), It->second);
    }
    return true;
  }

  // FIXME: Handle RecoveryExpr to try to hint some invalid calls.

private:
  using NameVec = SmallVector<StringRef, 8>;

  void processCall(Callee Callee, llvm::ArrayRef<const Expr *> Args) {
    assert(Callee.Decl || Callee.Loc);

    if (!Cfg.InlayHints.Parameters || Args.size() == 0)
      return;

    // The parameter name of a move or copy constructor is not very interesting.
    if (Callee.Decl)
      if (auto *Ctor = dyn_cast<CXXConstructorDecl>(Callee.Decl))
        if (Ctor->isCopyOrMoveConstructor())
          return;

    ArrayRef<const ParmVarDecl *> Params, ForwardedParams;
    // Resolve parameter packs to their forwarded parameter
    SmallVector<const ParmVarDecl *> ForwardedParamsStorage;
    if (Callee.Decl) {
      Params = maybeDropCxxExplicitObjectParameters(Callee.Decl->parameters());
      ForwardedParamsStorage = resolveForwardingParameters(Callee.Decl);
      ForwardedParams =
          maybeDropCxxExplicitObjectParameters(ForwardedParamsStorage);
    } else {
      Params = maybeDropCxxExplicitObjectParameters(Callee.Loc.getParams());
      ForwardedParams = {Params.begin(), Params.end()};
    }

    NameVec ParameterNames = chooseParameterNames(ForwardedParams);

    // Exclude setters (i.e. functions with one argument whose name begins with
    // "set"), and builtins like std::move/forward/... as their parameter name
    // is also not likely to be interesting.
    if (Callee.Decl &&
        (isSetter(Callee.Decl, ParameterNames) || isSimpleBuiltin(Callee.Decl)))
      return;

    for (size_t I = 0; I < ParameterNames.size() && I < Args.size(); ++I) {
      // Pack expansion expressions cause the 1:1 mapping between arguments and
      // parameters to break down, so we don't add further inlay hints if we
      // encounter one.
      if (isa<PackExpansionExpr>(Args[I])) {
        break;
      }

      StringRef Name = ParameterNames[I];
      bool NameHint = shouldHintName(Args[I], Name);
      bool ReferenceHint = shouldHintReference(Params[I], ForwardedParams[I]);

      if (NameHint || ReferenceHint) {
        addInlayHint(Args[I]->getSourceRange(), HintSide::Left,
                     InlayHintKind::Parameter, ReferenceHint ? "&" : "",
                     NameHint ? Name : "", ": ");
      }
    }
  }

  static bool isSetter(const FunctionDecl *Callee, const NameVec &ParamNames) {
    if (ParamNames.size() != 1)
      return false;

    StringRef Name = getSimpleName(*Callee);
    if (!Name.starts_with_insensitive("set"))
      return false;

    // In addition to checking that the function has one parameter and its
    // name starts with "set", also check that the part after "set" matches
    // the name of the parameter (ignoring case). The idea here is that if
    // the parameter name differs, it may contain extra information that
    // may be useful to show in a hint, as in:
    //   void setTimeout(int timeoutMillis);
    // This currently doesn't handle cases where params use snake_case
    // and functions don't, e.g.
    //   void setExceptionHandler(EHFunc exception_handler);
    // We could improve this by replacing `equals_insensitive` with some
    // `sloppy_equals` which ignores case and also skips underscores.
    StringRef WhatItIsSetting = Name.substr(3).ltrim("_");
    return WhatItIsSetting.equals_insensitive(ParamNames[0]);
  }

  // Checks if the callee is one of the builtins
  // addressof, as_const, forward, move(_if_noexcept)
  static bool isSimpleBuiltin(const FunctionDecl *Callee) {
    switch (Callee->getBuiltinID()) {
    case Builtin::BIaddressof:
    case Builtin::BIas_const:
    case Builtin::BIforward:
    case Builtin::BImove:
    case Builtin::BImove_if_noexcept:
      return true;
    default:
      return false;
    }
  }

  bool shouldHintName(const Expr *Arg, StringRef ParamName) {
    if (ParamName.empty())
      return false;

    // If the argument expression is a single name and it matches the
    // parameter name exactly, omit the name hint.
    if (ParamName == getSpelledIdentifier(Arg))
      return false;

    // Exclude argument expressions preceded by a /*paramName*/.
    if (isPrecededByParamNameComment(Arg, ParamName))
      return false;

    return true;
  }

  bool shouldHintReference(const ParmVarDecl *Param,
                           const ParmVarDecl *ForwardedParam) {
    // We add a & hint only when the argument is passed as mutable reference.
    // For parameters that are not part of an expanded pack, this is
    // straightforward. For expanded pack parameters, it's likely that they will
    // be forwarded to another function. In this situation, we only want to add
    // the reference hint if the argument is actually being used via mutable
    // reference. This means we need to check
    // 1. whether the value category of the argument is preserved, i.e. each
    //    pack expansion uses std::forward correctly.
    // 2. whether the argument is ever copied/cast instead of passed
    //    by-reference
    // Instead of checking this explicitly, we use the following proxy:
    // 1. the value category can only change from rvalue to lvalue during
    //    forwarding, so checking whether both the parameter of the forwarding
    //    function and the forwarded function are lvalue references detects such
    //    a conversion.
    // 2. if the argument is copied/cast somewhere in the chain of forwarding
    //    calls, it can only be passed on to an rvalue reference or const lvalue
    //    reference parameter. Thus if the forwarded parameter is a mutable
    //    lvalue reference, it cannot have been copied/cast to on the way.
    // Additionally, we should not add a reference hint if the forwarded
    // parameter was only partially resolved, i.e. points to an expanded pack
    // parameter, since we do not know how it will be used eventually.
    auto Type = Param->getType();
    auto ForwardedType = ForwardedParam->getType();
    return Type->isLValueReferenceType() &&
           ForwardedType->isLValueReferenceType() &&
           !ForwardedType.getNonReferenceType().isConstQualified() &&
           !isExpandedFromParameterPack(ForwardedParam);
  }

  // Checks if "E" is spelled in the main file and preceded by a C-style comment
  // whose contents match ParamName (allowing for whitespace and an optional "="
  // at the end.
  bool isPrecededByParamNameComment(const Expr *E, StringRef ParamName) {
    auto &SM = AST.getSourceManager();
    auto FileLoc = SM.getFileLoc(E->getBeginLoc());
    auto Decomposed = SM.getDecomposedLoc(FileLoc);
    if (Decomposed.first != MainFileID)
      return false;

    StringRef SourcePrefix = MainFileBuf.substr(0, Decomposed.second);
    // Allow whitespace between comment and expression.
    SourcePrefix = SourcePrefix.rtrim();
    // Check for comment ending.
    if (!SourcePrefix.consume_back("*/"))
      return false;
    // Ignore some punctuation and whitespace around comment.
    // In particular this allows designators to match nicely.
    llvm::StringLiteral IgnoreChars = " =.";
    SourcePrefix = SourcePrefix.rtrim(IgnoreChars);
    ParamName = ParamName.trim(IgnoreChars);
    // Other than that, the comment must contain exactly ParamName.
    if (!SourcePrefix.consume_back(ParamName))
      return false;
    SourcePrefix = SourcePrefix.rtrim(IgnoreChars);
    return SourcePrefix.ends_with("/*");
  }

  // If "E" spells a single unqualified identifier, return that name.
  // Otherwise, return an empty string.
  static StringRef getSpelledIdentifier(const Expr *E) {
    E = E->IgnoreUnlessSpelledInSource();

    if (auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (!DRE->getQualifier())
        return getSimpleName(*DRE->getDecl());

    if (auto *ME = dyn_cast<MemberExpr>(E))
      if (!ME->getQualifier() && ME->isImplicitAccess())
        return getSimpleName(*ME->getMemberDecl());

    return {};
  }

  NameVec chooseParameterNames(ArrayRef<const ParmVarDecl *> Parameters) {
    NameVec ParameterNames;
    for (const auto *P : Parameters) {
      if (isExpandedFromParameterPack(P)) {
        // If we haven't resolved a pack paramater (e.g. foo(Args... args)) to a
        // non-pack parameter, then hinting as foo(args: 1, args: 2, args: 3) is
        // unlikely to be useful.
        ParameterNames.emplace_back();
      } else {
        auto SimpleName = getSimpleName(*P);
        // If the parameter is unnamed in the declaration:
        // attempt to get its name from the definition
        if (SimpleName.empty()) {
          if (const auto *PD = getParamDefinition(P)) {
            SimpleName = getSimpleName(*PD);
          }
        }
        ParameterNames.emplace_back(SimpleName);
      }
    }

    // Standard library functions often have parameter names that start
    // with underscores, which makes the hints noisy, so strip them out.
    for (auto &Name : ParameterNames)
      stripLeadingUnderscores(Name);

    return ParameterNames;
  }

  // for a ParmVarDecl from a function declaration, returns the corresponding
  // ParmVarDecl from the definition if possible, nullptr otherwise.
  static const ParmVarDecl *getParamDefinition(const ParmVarDecl *P) {
    if (auto *Callee = dyn_cast<FunctionDecl>(P->getDeclContext())) {
      if (auto *Def = Callee->getDefinition()) {
        auto I = std::distance(Callee->param_begin(),
                               llvm::find(Callee->parameters(), P));
        if (I < (int)Callee->getNumParams()) {
          return Def->getParamDecl(I);
        }
      }
    }
    return nullptr;
  }

  // We pass HintSide rather than SourceLocation because we want to ensure
  // it is in the same file as the common file range.
  void addInlayHint(SourceRange R, HintSide Side, InlayHintKind Kind,
                    llvm::StringRef Prefix, llvm::StringRef Label,
                    llvm::StringRef Suffix) {
    auto LSPRange = getHintRange(R);
    if (!LSPRange)
      return;

    addInlayHint(*LSPRange, Side, Kind, Prefix, Label, Suffix);
  }

  void addInlayHint(SourceRange R, HintSide Side, InlayHintKind Kind,
                    llvm::StringRef Prefix,
                    std::vector<InlayHintLabelPart> Labels,
                    llvm::StringRef Suffix) {
    auto LSPRange = getHintRange(R);
    if (!LSPRange)
      return;

    addInlayHint(*LSPRange, Side, Kind, Prefix, std::move(Labels), Suffix);
  }

  void addInlayHint(Range LSPRange, HintSide Side, InlayHintKind Kind,
                    llvm::StringRef Prefix, llvm::StringRef Label,
                    llvm::StringRef Suffix) {
    return addInlayHint(LSPRange, Side, Kind, Prefix,
                        /*Labels=*/std::vector<InlayHintLabelPart>{Label.str()},
                        Suffix);
  }

  void addInlayHint(Range LSPRange, HintSide Side, InlayHintKind Kind,
                    llvm::StringRef Prefix,
                    std::vector<InlayHintLabelPart> Labels,
                    llvm::StringRef Suffix) {
    assert(!Labels.empty() && "Expected non-empty labels");
    // We shouldn't get as far as adding a hint if the category is disabled.
    // We'd like to disable as much of the analysis as possible above instead.
    // Assert in debug mode but add a dynamic check in production.
    assert(Cfg.InlayHints.Enabled && "Shouldn't get here if disabled!");
    switch (Kind) {
#define CHECK_KIND(Enumerator, ConfigProperty)                                 \
  case InlayHintKind::Enumerator:                                              \
    assert(Cfg.InlayHints.ConfigProperty &&                                    \
           "Shouldn't get here if kind is disabled!");                         \
    if (!Cfg.InlayHints.ConfigProperty)                                        \
      return;                                                                  \
    break
      CHECK_KIND(Parameter, Parameters);
      CHECK_KIND(Type, DeducedTypes);
      CHECK_KIND(Designator, Designators);
      CHECK_KIND(BlockEnd, BlockEnd);
#undef CHECK_KIND
    }

    Position LSPPos = Side == HintSide::Left ? LSPRange.start : LSPRange.end;
    if (RestrictRange &&
        (LSPPos < RestrictRange->start || !(LSPPos < RestrictRange->end)))
      return;
    bool PadLeft = Prefix.consume_front(" ");
    bool PadRight = Suffix.consume_back(" ");
    if (!Prefix.empty()) {
      if (auto &Label = Labels.front(); !Label.location)
        Label.value = Prefix.str() + Label.value;
      else
        Labels.insert(Labels.begin(), InlayHintLabelPart(Prefix.str()));
    }
    if (!Suffix.empty()) {
      if (auto &Label = Labels.back(); !Label.location)
        Label.value += Suffix.str();
      else
        Labels.push_back(InlayHintLabelPart(Suffix.str()));
    }
    Results.push_back(InlayHint{LSPPos,
                                /*label=*/std::move(Labels), Kind, PadLeft,
                                PadRight, LSPRange});
  }

  // Get the range of the main file that *exactly* corresponds to R.
  std::optional<Range> getHintRange(SourceRange R) {
    const auto &SM = AST.getSourceManager();
    auto Spelled = Tokens.spelledForExpanded(Tokens.expandedTokens(R));
    // TokenBuffer will return null if e.g. R corresponds to only part of a
    // macro expansion.
    if (!Spelled || Spelled->empty())
      return std::nullopt;
    // Hint must be within the main file, not e.g. a non-preamble include.
    if (SM.getFileID(Spelled->front().location()) != SM.getMainFileID() ||
        SM.getFileID(Spelled->back().location()) != SM.getMainFileID())
      return std::nullopt;
    return Range{sourceLocToPosition(SM, Spelled->front().location()),
                 sourceLocToPosition(SM, Spelled->back().endLocation())};
  }

  std::vector<InlayHintLabelPart> buildTypeHint(QualType T,
                                                llvm::StringRef Prefix) {
    TypeHintBuilder Builder(AST, MainFilePath, TypeHintPolicy, Prefix);
    Builder.VisitQualType(T, /*AppendSpaceToTopLevelQuals=*/false);
    return Builder.take();
  }

  void addTypeHint(SourceRange R, QualType T, llvm::StringRef Prefix) {
    if (!Cfg.InlayHints.DeducedTypes || T.isNull())
      return;

    // The sugared type is more useful in some cases, and the canonical
    // type in other cases.
    auto Desugared = maybeDesugar(AST, T);
    auto Chunks = buildTypeHint(Desugared, Prefix);
    if (T != Desugared) {
      if (shouldPrintTypeHint(Chunks)) {
        addInlayHint(R, HintSide::Right, InlayHintKind::Type,
                     /*Prefix=*/"", // We have handled prefixes in the builder.
                     std::move(Chunks),
                     /*Suffix=*/"");
        return;
      }
      // If the desugared type is too long to display, fallback to the sugared
      // type.
      Chunks = buildTypeHint(T, Prefix);
    }
    if (shouldPrintTypeHint(Chunks))
      addInlayHint(R, HintSide::Right, InlayHintKind::Type,
                   /*Prefix=*/"", // We have handled prefixes in the builder.
                   std::move(Chunks),
                   /*Suffix=*/"");
  }

  void addDesignatorHint(SourceRange R, llvm::StringRef Text) {
    addInlayHint(R, HintSide::Left, InlayHintKind::Designator,
                 /*Prefix=*/"", Text, /*Suffix=*/"=");
  }

  bool shouldPrintTypeHint(
      llvm::ArrayRef<InlayHintLabelPart> TypeLabels) const noexcept {
    return Cfg.InlayHints.TypeNameLimit == 0 ||
           lengthOfInlayHintLabel(TypeLabels) < Cfg.InlayHints.TypeNameLimit;
  }

  void addBlockEndHint(SourceRange BraceRange, StringRef DeclPrefix,
                       StringRef Name, StringRef OptionalPunctuation) {
    auto HintRange = computeBlockEndHintRange(BraceRange, OptionalPunctuation);
    if (!HintRange)
      return;

    std::string Label = DeclPrefix.str();
    if (!Label.empty() && !Name.empty())
      Label += ' ';
    Label += Name;

    constexpr unsigned HintMaxLengthLimit = 60;
    if (Label.length() > HintMaxLengthLimit)
      return;

    addInlayHint(*HintRange, HintSide::Right, InlayHintKind::BlockEnd, " // ",
                 Label, "");
  }

  // Compute the LSP range to attach the block end hint to, if any allowed.
  // 1. "}" is the last non-whitespace character on the line. The range of "}"
  // is returned.
  // 2. After "}", if the trimmed trailing text is exactly
  // `OptionalPunctuation`, say ";". The range of "} ... ;" is returned.
  // Otherwise, the hint shouldn't be shown.
  std::optional<Range> computeBlockEndHintRange(SourceRange BraceRange,
                                                StringRef OptionalPunctuation) {
    constexpr unsigned HintMinLineLimit = 2;

    auto &SM = AST.getSourceManager();
    auto [BlockBeginFileId, BlockBeginOffset] =
        SM.getDecomposedLoc(SM.getFileLoc(BraceRange.getBegin()));
    auto RBraceLoc = SM.getFileLoc(BraceRange.getEnd());
    auto [RBraceFileId, RBraceOffset] = SM.getDecomposedLoc(RBraceLoc);

    // Because we need to check the block satisfies the minimum line limit, we
    // require both source location to be in the main file. This prevents hint
    // to be shown in weird cases like '{' is actually in a "#include", but it's
    // rare anyway.
    if (BlockBeginFileId != MainFileID || RBraceFileId != MainFileID)
      return std::nullopt;

    StringRef RestOfLine = MainFileBuf.substr(RBraceOffset).split('\n').first;
    if (!RestOfLine.starts_with("}"))
      return std::nullopt;

    StringRef TrimmedTrailingText = RestOfLine.drop_front().trim();
    if (!TrimmedTrailingText.empty() &&
        TrimmedTrailingText != OptionalPunctuation)
      return std::nullopt;

    auto BlockBeginLine = SM.getLineNumber(BlockBeginFileId, BlockBeginOffset);
    auto RBraceLine = SM.getLineNumber(RBraceFileId, RBraceOffset);

    // Don't show hint on trivial blocks like `class X {};`
    if (BlockBeginLine + HintMinLineLimit - 1 > RBraceLine)
      return std::nullopt;

    // This is what we attach the hint to, usually "}" or "};".
    StringRef HintRangeText = RestOfLine.take_front(
        TrimmedTrailingText.empty()
            ? 1
            : TrimmedTrailingText.bytes_end() - RestOfLine.bytes_begin());

    Position HintStart = sourceLocToPosition(SM, RBraceLoc);
    Position HintEnd = sourceLocToPosition(
        SM, RBraceLoc.getLocWithOffset(HintRangeText.size()));
    return Range{HintStart, HintEnd};
  }

  static bool isFunctionObjectCallExpr(CallExpr *E) noexcept {
    if (auto *CallExpr = dyn_cast<CXXOperatorCallExpr>(E))
      return CallExpr->getOperator() == OverloadedOperatorKind::OO_Call;
    return false;
  }

  std::vector<InlayHint> &Results;
  ASTContext &AST;
  const syntax::TokenBuffer &Tokens;
  const Config &Cfg;
  std::optional<Range> RestrictRange;
  FileID MainFileID;
  StringRef MainFileBuf;
  StringRef MainFilePath;
  const HeuristicResolver *Resolver;
  PrintingPolicy TypeHintPolicy;
};

} // namespace

std::vector<InlayHint> inlayHints(ParsedAST &AST,
                                  std::optional<Range> RestrictRange) {
  std::vector<InlayHint> Results;
  const auto &Cfg = Config::current();
  if (!Cfg.InlayHints.Enabled)
    return Results;
  InlayHintVisitor Visitor(Results, AST, Cfg, std::move(RestrictRange));
  Visitor.TraverseAST(AST.getASTContext());

  // De-duplicate hints. Duplicates can sometimes occur due to e.g. explicit
  // template instantiations.
  llvm::sort(Results);
  Results.erase(std::unique(Results.begin(), Results.end()), Results.end());

  return Results;
}

} // namespace clangd
} // namespace clang
