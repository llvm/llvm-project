//=== SemaEval.cpp - Sema handling of effectual constant evaluation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Sema handling of language-mandated constant evaluation,
// which can perform actions and queries against the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/SemaProxy.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaInternal.h"

using namespace clang;

namespace {

class SemaProxyImpl : public clang::SemaProxy {
public:
  SemaProxyImpl(clang::Sema &SemaRef) : SemaRef(SemaRef) {}

  void instantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                     FunctionDecl *Function) override;

private:
  Sema &SemaRef;
};

void SemaProxyImpl::instantiateFunctionDefinition(
    SourceLocation PointOfInstantiation, FunctionDecl *Function) {
  SemaRef.InstantiateFunctionDefinition(
        PointOfInstantiation, Function, /*Recursive=*/true,
        /*DefinitionRequired=*/true, /*AtEndOfTU=*/false);
}

} // anonymous namespace

namespace clang {

SemaProxy *Sema::makeProxyForEval(Sema &SemaRef) {
  return new SemaProxyImpl(SemaRef);
}
} // end namespace clang
