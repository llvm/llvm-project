//===--- SymbolOperation.cpp - --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/SymbolOperation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Tooling/Refactor/RefactoringActionFinder.h"

using namespace clang;

/// Return true if the given local record decl escapes the given enclosing
/// function or block \p Ctx.
static bool escapesEnclosingDecl(const DeclContext *Ctx, const RecordDecl *RD) {
  QualType ReturnType;
  bool DependentBlock = false;
  if (const auto *FD = dyn_cast<FunctionDecl>(Ctx))
    ReturnType = FD->getReturnType();
  else if (const auto *BD = dyn_cast<BlockDecl>(Ctx)) {
    ReturnType = BD->getSignatureAsWritten()->getType();
    // Blocks that don't have an explicitly specified type (represented with a
    // dependent type) could potentially return the record, e.g.
    // auto block = ^ {
    //   struct Foo { };
    //   return Foo();
    // };
    if (const auto *FT = ReturnType->getAs<FunctionType>())
      ReturnType = FT->getReturnType();
    if (ReturnType->isDependentType())
      DependentBlock = true;
  } else
    return false;

  // The record can be returned from its enclosing function when the function's
  // return type is auto.
  //
  // FIXME: Use a smarter heuristic that detects if the record type is
  // actually returned from the function. Have to account for inner records,
  // like in the example below:
  //
  //   auto foo() {
  //     struct Foo { struct Bar { }; };
  //     return Foo::Bar();
  //   };
  //
  // for types that depend on the record, like in the example below:
  //
  //   auto foo() {
  //     template<typename T> struct C<T> { T x; };
  //     struct Foo { struct Bar { }; };
  //     return C<Bar>();
  //   }
  //
  // and for things like typedefs and function types as well.
  if (!DependentBlock && !ReturnType->getContainedAutoType())
    return false;

  // Even if the enclosing function returns the local record, this record is
  // still local if the enclosing function is inside a function/method that
  // doesn't return this record.
  const auto *D = cast<Decl>(Ctx);
  if (D->isLexicallyWithinFunctionOrMethod())
    return escapesEnclosingDecl(D->getParentFunctionOrMethod(), RD);

  return true;
}

static bool escapesEnclosingDecl(const RecordDecl *RD,
                                 const LangOptions &LangOpts) {
  // We only care about things that escape in header files since things that
  // escape in source files will be used only in the initial TU.
  return LangOpts.IsHeaderFile &&
         escapesEnclosingDecl(RD->getParentFunctionOrMethod(), RD);
}

/// Return true if the given declaration corresponds to a local symbol.
bool clang::tooling::isLocalSymbol(const NamedDecl *FoundDecl,
                                   const LangOptions &LangOpts) {
  // Template parameters aren't indexed, so use local rename.
  if (isa<TemplateTypeParmDecl>(FoundDecl) ||
      isa<NonTypeTemplateParmDecl>(FoundDecl) ||
      isa<TemplateTemplateParmDecl>(FoundDecl))
    return true;

  if (const auto *VD = dyn_cast<VarDecl>(FoundDecl))
    return VD->isLocalVarDeclOrParm();

  // Objective-C selector renames must be global.
  if (isa<ObjCMethodDecl>(FoundDecl))
    return false;

  // Local declarations are defined in a function or a method, or are anonymous.
  if (!FoundDecl->isLexicallyWithinFunctionOrMethod())
    return false;

  // A locally defined record is global when it is returned from the enclosing
  // function because we can refer to its destructor externally.
  if (const auto *RD = dyn_cast<CXXRecordDecl>(FoundDecl))
    return !escapesEnclosingDecl(RD, LangOpts);

  // A locally defined field is global when its record is returned from the
  // enclosing function.
  if (const auto *FD = dyn_cast<FieldDecl>(FoundDecl))
    return !escapesEnclosingDecl(FD->getParent(), LangOpts);

  if (const auto *MD = dyn_cast<CXXMethodDecl>(FoundDecl)) {
    // A locally defined method is global when its record is returned from the
    // enclosing function.
    if (escapesEnclosingDecl(MD->getParent(), LangOpts))
      return false;

    // Method renames can be local only iff this method doesn't override
    // a global method, for example:
    //
    //   void func() {
    //     struct Foo: GlobalSuper {
    //       // When renaming foo we should also rename GlobalSuper's foo
    //       void foo() override;
    //     }
    //   }
    //
    // FIXME: We can try to be smarter about it and check if we override
    // a local method, which would make this method local as well.
    return !MD->isVirtual();
  }

  return true;
}

static const NamedDecl *
findDeclThatRequiresImplementationTU(const NamedDecl *FoundDecl) {
  // TODO: implement the rest.
  if (const ObjCIvarDecl *IVarDecl = dyn_cast<ObjCIvarDecl>(FoundDecl)) {
    // We need the implementation TU when the IVAR is declared in an @interface
    // without an @implementation.
    if (const auto *ID =
            dyn_cast<ObjCInterfaceDecl>(IVarDecl->getDeclContext())) {
      if (!ID->getImplementation())
        return IVarDecl;
    }
  }
  return nullptr;
}

namespace clang {
namespace tooling {

SymbolOperation::SymbolOperation(const NamedDecl *FoundDecl,
                                 ASTContext &Context)
    : IsLocal(isLocalSymbol(FoundDecl, Context.getLangOpts())) {
  // Take the category declaration if this is a category implementation.
  if (const auto *CategoryImplDecl =
          dyn_cast<ObjCCategoryImplDecl>(FoundDecl)) {
    if (const auto *CategoryDecl = CategoryImplDecl->getCategoryDecl())
      FoundDecl = CategoryDecl;
  }
  // Use the property if this method is a getter/setter.
  else if (const auto *MethodDecl = dyn_cast<ObjCMethodDecl>(FoundDecl)) {
    if (const auto *PropertyDecl =
            MethodDecl->getCanonicalDecl()->findPropertyDecl()) {
      // Don't use the property if the getter/setter method has an explicitly
      // specified name.
      if (MethodDecl->param_size() == 0
              ? !PropertyDecl->hasExplicitGetterName()
              : !PropertyDecl->hasExplicitSetterName())
        FoundDecl = PropertyDecl;
    }
  }

  DeclThatRequiresImplementationTU =
      findDeclThatRequiresImplementationTU(FoundDecl);

  // TODO: Split into initiation that works after implementation TU is loaded.

  // Find the set of symbols that this operation has to work on.
  auto AddSymbol = [this, &Context](const NamedDecl *FoundDecl) {
    unsigned Index = Symbols.size();
    Symbols.push_back(rename::Symbol(FoundDecl, Index, Context.getLangOpts()));
    for (const auto &USR : findSymbolsUSRSet(FoundDecl, Context))
      USRToSymbol.insert(std::make_pair(USR.getKey(), Index));
  };
  AddSymbol(FoundDecl);
  // Take getters, setters and ivars into account when dealing with
  // Objective-C @property declarations.
  if (const auto *PropertyDecl = dyn_cast<ObjCPropertyDecl>(FoundDecl)) {
    // FIXME: findSymbolsUSRSet is called for every symbol we add, which is
    // inefficient since we currently have to traverse the AST every time it is
    // called. Fix this so that the AST isn't traversed more than once.
    if (!PropertyDecl->hasExplicitGetterName()) {
      if (const auto *Getter = PropertyDecl->getGetterMethodDecl())
        AddSymbol(Getter);
    }
    if (!PropertyDecl->hasExplicitSetterName()) {
      if (const auto *Setter = PropertyDecl->getSetterMethodDecl())
        AddSymbol(Setter);
    }
  }
}

} // end namespace tooling
} // end namespace clang
