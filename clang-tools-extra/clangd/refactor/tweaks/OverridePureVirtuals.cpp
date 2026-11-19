//===--- OverridePureVirtuals.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tweak to automatically generate stubs for pure virtual methods inherited from
// base classes.
//
// Purpose:
//   - Simplifies making a derived class concrete by automating the creation of
//     required method overrides from abstract bases.
//
// Tweak Summary:
//
// 1. Activation Conditions (prepare):
//    - The tweak activates when the cursor is over a C++ class definition.
//    - The class must be abstract (it, or its base classes, have unimplemented
//      pure virtual functions).
//    - It must also inherit from at least one other abstract class.
//
// 2. Identifying Missing Methods:
//    - The tweak scans the inheritance hierarchy of the current class.
//    - It identifies all unique pure virtual methods from base classes
//      that are not yet implemented or overridden.
//    - These missing methods are then grouped by their original access
//      specifier (e.g., public, protected).
//
// 3. Code Generation and Insertion:
//    - For each group of missing methods, stubs are inserted.
//    - If an access specifier section (like `public:`) exists, stubs are
//      inserted there; otherwise, a new section is created and appended.
//    - Each generated stub includes the `override` keyword, a `// TODO:`
//      comment, and a `static_assert(false, ...)` to force a compile-time
//      error if the method remains unimplemented.
//    - The base method's signature is adjusted for the override (e.g.,
//      `virtual` and `= 0` are removed, and all nested name specifiers
//      that do not enclose the derived class are added to the types of
//      parameters and return value).
//
// 4. Code Action Provided:
//    - A single code action titled "Override pure virtual methods" is offered.
//    - Applying this action results in a single source file modification
//      containing all the generated method stubs.
//
// Example:
//
//  class Base {
//  public:
//    virtual void publicMethod() = 0;
//  protected:
//    virtual auto privateMethod() const -> int = 0;
//  };
//
// Before:
//                              // cursor here
//  class Derived : public Base {}^;
//
// After:
//
// class Derived : public Base {
//  public:
//   void publicMethod() override {
//     // TODO: Implement this pure virtual method.
//     static_assert(false, "Method `publicMethod` is not implemented.");
//   }
//
//  protected:
//   auto privateMethod() const -> int override {
//     // TODO: Implement this pure virtual method.
//     static_assert(false, "Method `privateMethod` is not implemented.");
//   }
// };
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/AttributeCommonInfo.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace clang {
namespace clangd {
namespace {

class OverridePureVirtuals final : public Tweak {
public:
  const char *id() const final; // defined by REGISTER_TWEAK.
  bool prepare(const Selection &Sel) override;
  Expected<Effect> apply(const Selection &Sel) override;
  std::string title() const override { return "Override pure virtual methods"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::QUICKFIX_KIND;
  }

private:
  // Stores the CXXRecordDecl of the class being modified.
  const CXXRecordDecl *CurrentDeclDef = nullptr;
  // Stores pure virtual methods that need overriding, grouped by their original
  // access specifier.
  llvm::MapVector<AccessSpecifier, llvm::SmallVector<const CXXMethodDecl *>>
      MissingMethodsByAccess;
  // Stores the source locations of existing access specifiers in CurrentDecl.
  llvm::MapVector<AccessSpecifier, SourceLocation> AccessSpecifierLocations;
  // Helper function to gather information before applying the tweak.
  void collectMissingPureVirtuals();
};

REGISTER_TWEAK(OverridePureVirtuals)

// Function to get all unique pure virtual methods from the entire
// base class hierarchy of CurrentDeclDef.
llvm::SmallVector<const clang::CXXMethodDecl *>
getAllUniquePureVirtualsFromBaseHierarchy(
    const clang::CXXRecordDecl *CurrentDeclDef) {
  llvm::SmallVector<const clang::CXXMethodDecl *> AllPureVirtualsInHierarchy;
  llvm::DenseSet<const clang::CXXMethodDecl *> CanonicalPureVirtualsSeen;

  if (!CurrentDeclDef || !CurrentDeclDef->getDefinition())
    return AllPureVirtualsInHierarchy;

  const clang::CXXRecordDecl *Def = CurrentDeclDef->getDefinition();

  Def->forallBases([&](const clang::CXXRecordDecl *BaseDefinition) {
    for (const clang::CXXMethodDecl *Method : BaseDefinition->methods()) {
      if (Method->isPureVirtual() &&
          CanonicalPureVirtualsSeen.insert(Method->getCanonicalDecl()).second)
        AllPureVirtualsInHierarchy.emplace_back(Method);
    }
    // Continue iterating through all bases.
    return true;
  });

  return AllPureVirtualsInHierarchy;
}

// Gets canonical declarations of methods already overridden or implemented in
// class D.
llvm::SetVector<const CXXMethodDecl *>
getImplementedOrOverriddenCanonicals(const CXXRecordDecl *D) {
  llvm::SetVector<const CXXMethodDecl *> ImplementedSet;
  for (const CXXMethodDecl *M : D->methods()) {
    // If M provides an implementation for any virtual method it overrides.
    // A method is an "implementation" if it's virtual and not pure.
    // Or if it directly overrides a base method.
    for (const CXXMethodDecl *OverriddenM : M->overridden_methods())
      ImplementedSet.insert(OverriddenM->getCanonicalDecl());
  }
  return ImplementedSet;
}

// Get the location of every colon of the `AccessSpecifier`.
llvm::MapVector<AccessSpecifier, SourceLocation>
getSpecifierLocations(const CXXRecordDecl *D) {
  llvm::MapVector<AccessSpecifier, SourceLocation> Locs;
  for (auto *DeclNode : D->decls()) {
    if (const auto *ASD = llvm::dyn_cast<AccessSpecDecl>(DeclNode))
      Locs[ASD->getAccess()] = ASD->getColonLoc();
  }
  return Locs;
}

bool hasAbstractBaseAncestor(const clang::CXXRecordDecl *CurrentDecl) {
  assert(CurrentDecl && CurrentDecl->getDefinition());

  return llvm::any_of(
      CurrentDecl->getDefinition()->bases(), [](CXXBaseSpecifier BaseSpec) {
        const auto *D = BaseSpec.getType()->getAsCXXRecordDecl();
        const auto *Def = D ? D->getDefinition() : nullptr;
        return Def && Def->isAbstract();
      });
}

// The tweak is available if the selection is over an abstract C++ class
// definition that also inherits from at least one other abstract class.
bool OverridePureVirtuals::prepare(const Selection &Sel) {
  const SelectionTree::Node *Node = Sel.ASTSelection.commonAncestor();
  if (!Node)
    return false;

  // Make sure we have a definition.
  CurrentDeclDef = Node->ASTNode.get<CXXRecordDecl>();
  if (!CurrentDeclDef || !CurrentDeclDef->getDefinition())
    return false;

  // From now on, we should work with the definition.
  CurrentDeclDef = CurrentDeclDef->getDefinition();

  // Only offer for abstract classes with abstract bases.
  return CurrentDeclDef->isAbstract() &&
         hasAbstractBaseAncestor(CurrentDeclDef);
}

// Collects all pure virtual methods from base classes that `CurrentDeclDef` has
// not yet overridden, grouped by their original access specifier.
//
// Results are stored in `MissingMethodsByAccess` and `AccessSpecifierLocations`
// is also populated.
void OverridePureVirtuals::collectMissingPureVirtuals() {
  if (!CurrentDeclDef)
    return;

  AccessSpecifierLocations = getSpecifierLocations(CurrentDeclDef);
  MissingMethodsByAccess.clear();

  // Get all unique pure virtual methods from the entire base class hierarchy.
  llvm::SmallVector<const CXXMethodDecl *> AllPureVirtualsInHierarchy =
      getAllUniquePureVirtualsFromBaseHierarchy(CurrentDeclDef);

  // Get methods already implemented or overridden in CurrentDecl.
  const auto ImplementedOrOverriddenSet =
      getImplementedOrOverriddenCanonicals(CurrentDeclDef);

  // Filter AllPureVirtualsInHierarchy to find those not in
  // ImplementedOrOverriddenSet, which needs to be overriden.
  for (const CXXMethodDecl *BaseMethod : AllPureVirtualsInHierarchy) {
    bool AlreadyHandled = ImplementedOrOverriddenSet.contains(BaseMethod);
    if (!AlreadyHandled)
      MissingMethodsByAccess[BaseMethod->getAccess()].emplace_back(BaseMethod);
  }
}

std::string generateOverrideString(const CXXRecordDecl *CurrentDeclDef,
                                   const CXXMethodDecl *Method) {
  // Parts of this function mirror DeclPrinter::VisitFunctionDecl in
  // (clang/lib/AST/DeclPrinter.cpp). We can't simply use DeclPrinter, because
  // Policy.FullyQualifiedName qualifies the types and the method name together,
  // and the name of the override has to stay unqualified.

  std::string MethodDecl;
  auto OS = llvm::raw_string_ostream(MethodDecl);
  std::string Pragmas;
  auto POS = llvm::raw_string_ostream(Pragmas);

  // Create policy that prints all nested name specifiers that do not enclose
  // the class to be modified.
  PrintingPolicy PP = Method->getASTContext().getPrintingPolicy();
  PP.FullyQualifiedName = true;

  class PrintCB : public PrintingCallbacks {
  public:
    PrintCB(const DeclContext *CurContext) : CurContext(CurContext) {}
    virtual ~PrintCB() {}
    bool isScopeVisible(const DeclContext *DC) const override {
      return DC->Encloses(CurContext);
    }

  private:
    const DeclContext *CurContext;
  };

  PrintCB PCB(CurrentDeclDef);
  PP.Callbacks = &PCB;

  // Get function prototype
  QualType Ty = Method->getType();
  const FunctionProtoType *FT = Ty->getAs<FunctionProtoType>();

  // Attribute specifiers
  if (Method->hasAttrs()) {
    for (const Attr *A : Method->attrs()) {
      // Pragmas will be printed above the function header. All other attributes
      // are printed at the beginning of the header.
      if (A->getSyntax() == AttributeCommonInfo::AS_Pragma) {
        A->printPretty(POS, PP);
        continue;
      }
      // Keyword attributes are not copied: `override` is added explicitly
      // below, `final` is intentionally not carried over, and an asm label has
      // to follow the declarator, so it can't be placed in front like the other
      // attributes.
      if (A->isInherited() || A->isImplicit() || A->isKeywordAttribute())
        continue;

      A->printPretty(OS, PP);
      OS << " ";
    }
  }

  // Relevant specifiers
  if (Method->isInlineSpecified())
    OS << "inline ";
  if (Method->isConstexprSpecified())
    OS << "constexpr ";
  if (Method->isConsteval())
    OS << "consteval ";

  // Return type
  bool HasReturnType = !isa<CXXConversionDecl>(Method);

  if (HasReturnType) {
    if (FT->hasTrailingReturn())
      OS << "auto ";
    else {
      FT->getReturnType().print(OS, PP);
      OS << " ";
    }
  }

  // Calling convention
  CallingConv CC = FT->getCallConv();
  // An override must have the same convention as the method it overrides,
  // otherwise Clang rejects it ("different calling convention attributes").
  // Only conventions that are spelled as a plain attribute name are handled.
  switch (CC) {
  case CC_X86StdCall:
  case CC_X86FastCall:
  case CC_X86ThisCall:
  case CC_X86VectorCall:
  case CC_X86RegCall:
  case CC_Win64:
  case CC_X86_64SysV:
  case CC_PreserveMost:
  case CC_PreserveAll:
  case CC_PreserveNone:
    OS << "__attribute__((" << FunctionType::getNameForCallConv(CC) << ")) ";
    break;
  default:
    // Not copied: CC_C is the default and printing it would only add noise
    // (or an ignored-attribute warning on some targets), some conventions are
    // unlikely to occur on C++ methods, and some (e.g. pcs("..."),
    // riscv_vls_cc(N)) need an argument that we do not print.
    break;
  }

  // Unqualified method name
  OS << Method->getNameAsString();

  // Parameter list
  OS << "(";
  llvm::ListSeparator LS;
  for (unsigned I = 0, E = Method->getNumParams(); I < E; ++I) {
    OS << LS;
    Method->getParamDecl(I)->print(OS, PP);
  }
  if (FT->isVariadic()) {
    if (Method->getNumParams())
      OS << ", ";
    OS << "...";
  }
  OS << ")";

  // CV Qualifiers
  if (FT->isConst())
    OS << " const";
  if (FT->isVolatile())
    OS << " volatile";

  // Reference Qualifiers
  switch (FT->getRefQualifier()) {
  case RQ_None:
    break;
  case RQ_LValue:
    OS << " &";
    break;
  case RQ_RValue:
    OS << " &&";
    break;
  }

  // Exception specifier
  if (FT->hasDynamicExceptionSpec()) {
    OS << " throw(";
    if (FT->getExceptionSpecType() == EST_MSAny)
      OS << "...";
    else {
      llvm::ListSeparator LS;
      for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
        OS << LS;
        OS << FT->getExceptionType(I).getAsString(PP);
      }
    }
    OS << ")";
  } else if (isNoexceptExceptionSpec(FT->getExceptionSpecType())) {
    OS << " noexcept";
    if (isComputedNoexcept(FT->getExceptionSpecType())) {
      OS << "(";
      FT->getNoexceptExpr()->printPretty(OS, nullptr, PP);
      OS << ")";
    }
  }

  // Trailing return type
  if (HasReturnType && FT->hasTrailingReturn()) {
    OS << " -> ";
    FT->getReturnType().print(OS, PP);
  }

  return llvm::formatv(
             "\n{0}  {1} override {{\n"
             "    // TODO: Implement this pure virtual method.\n"
             "    static_assert(false, \"Method `{2}` is not implemented.\");\n"
             "  }",
             Pragmas, MethodDecl, Method->getNameAsString())
      .str();
}

// Free function to generate the string for a group of method overrides.
std::string generateOverridesStringForGroup(
    const CXXRecordDecl *CurrentDeclDef,
    llvm::SmallVector<const CXXMethodDecl *> Methods) {
  llvm::SmallVector<std::string> MethodsString;
  MethodsString.reserve(Methods.size());

  for (const CXXMethodDecl *Method : Methods) {
    MethodsString.emplace_back(generateOverrideString(CurrentDeclDef, Method));
  }

  return llvm::join(MethodsString, "\n") + '\n';
}

Expected<Tweak::Effect> OverridePureVirtuals::apply(const Selection &Sel) {
  // The correctness of this tweak heavily relies on the accurate population of
  // these members.
  collectMissingPureVirtuals();
  // The `prepare` should prevent this. If the prepare identifies an abstract
  // method, then is must have missing methods.
  assert(!MissingMethodsByAccess.empty());

  const auto &SM = Sel.AST->getSourceManager();

  tooling::Replacements EditReplacements;
  // Stores text for new access specifier sections that are not already present
  // in the class.
  // Example:
  //  public:    // ...
  //  protected: // ...
  std::string NewSectionsToAppendText;

  for (const auto &[AS, Methods] : MissingMethodsByAccess) {
    assert(!Methods.empty());

    std::string MethodsGroupString =
        generateOverridesStringForGroup(CurrentDeclDef, Methods);

    auto *ExistingSpecLocIter = AccessSpecifierLocations.find(AS);
    bool ASExists = ExistingSpecLocIter != AccessSpecifierLocations.end();
    if (ASExists) {
      // Access specifier section already exists in the class.
      // Get location immediately *after* the colon.
      SourceLocation InsertLoc =
          ExistingSpecLocIter->second.getLocWithOffset(1);

      // Create a replacement to insert the method declarations.
      // The replacement is at InsertLoc, has length 0 (insertion), and uses
      // InsertionText.
      std::string InsertionText = MethodsGroupString;
      tooling::Replacement Rep(SM, InsertLoc, 0, InsertionText);
      if (auto Err = EditReplacements.add(Rep))
        return llvm::Expected<Tweak::Effect>(std::move(Err));
    } else {
      // Access specifier section does not exist in the class.
      // These methods will be grouped into NewSectionsToAppendText and added
      // towards the end of the class definition.
      NewSectionsToAppendText +=
          getAccessSpelling(AS).str() + ':' + MethodsGroupString;
    }
  }

  // After processing all access specifiers, add any newly created sections
  // (stored in NewSectionsToAppendText) to the end of the class.
  if (!NewSectionsToAppendText.empty()) {
    // AppendLoc is the SourceLocation of the closing brace '}' of the class.
    // The replacement will insert text *before* this closing brace.
    SourceLocation AppendLoc = CurrentDeclDef->getBraceRange().getEnd();
    std::string FinalAppendText = std::move(NewSectionsToAppendText);

    if (!CurrentDeclDef->decls_empty() || !EditReplacements.empty()) {
      FinalAppendText = '\n' + FinalAppendText;
    }

    // Create a replacement to append the new sections.
    tooling::Replacement Rep(SM, AppendLoc, 0, FinalAppendText);
    if (auto Err = EditReplacements.add(Rep))
      return llvm::Expected<Tweak::Effect>(std::move(Err));
  }

  if (EditReplacements.empty()) {
    return llvm::make_error<llvm::StringError>(
        "No changes to apply (internal error or no methods generated).",
        llvm::inconvertibleErrorCode());
  }

  // Return the collected replacements as the effect of this tweak.
  return Effect::mainFileEdit(SM, EditReplacements);
}

} // namespace
} // namespace clangd
} // namespace clang
