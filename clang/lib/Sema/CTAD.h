//===--- CTAD.h - Helper functions for CTAD -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines helper functions for the class template argument deduction
//  (CTAD) implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

// Transform a given function parameter decl into a deduction guide parameter
// decl.
ParmVarDecl *transformFunctionTypeParam(
    Sema &SemaRef, ParmVarDecl *OldParam, DeclContext *DC,
    MultiLevelTemplateArgumentList &Args,
    llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs);

// Transform a given template type parameter into a deduction guide template
// parameter, rebuilding any internal references to earlier parameters and
// re-indexing as we go.
TemplateTypeParmDecl *transformTemplateTypeParam(
    Sema &SemaRef, DeclContext *DC, TemplateTypeParmDecl *TPT,
    MultiLevelTemplateArgumentList &Args, unsigned NewDepth, unsigned NewIndex);
// Similar to above, but for non-type template or template template parameters.
template <typename NonTypeTemplateOrTemplateTemplateParmDecl>
NonTypeTemplateOrTemplateTemplateParmDecl *
transformTemplateParam(Sema &SemaRef, DeclContext *DC,
                       NonTypeTemplateOrTemplateTemplateParmDecl *OldParam,
                       MultiLevelTemplateArgumentList &Args,
                       unsigned NewIndex) {
  // Ask the template instantiator to do the heavy lifting for us, then adjust
  // the index of the parameter once it's done.
  auto *NewParam = cast<NonTypeTemplateOrTemplateTemplateParmDecl>(
      SemaRef.SubstDecl(OldParam, DC, Args));
  NewParam->setPosition(NewIndex);
  return NewParam;
}

// Build a deduction guide with the specified parameter types.
FunctionTemplateDecl *buildDeductionGuide(
    Sema &SemaRef, TemplateDecl *OriginalTemplate,
    TemplateParameterList *TemplateParams, CXXConstructorDecl *Ctor,
    ExplicitSpecifier ES, TypeSourceInfo *TInfo, SourceLocation LocStart,
    SourceLocation Loc, SourceLocation LocEnd, bool IsImplicit,
    llvm::ArrayRef<TypedefNameDecl *> MaterializedTypedefs = {});

} // namespace clang
