//=== SemaProxy.cpp - Sema proxy for effectual constant evaluation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a proxy to the Sema class that can be provided to the
// constant evaluator, thereby facilitating evaluations capable of acting on and
// querying the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaProxy.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Sema.h"

namespace clang {
namespace sema {

void EvalProxy::instantiateFunctionDefinition(
    SourceLocation PointOfInstantiation, FunctionDecl *Function) {
  SemaRef.InstantiateFunctionDefinition(
      PointOfInstantiation, Function, /*Recursive=*/true,
      /*DefinitionRequired=*/true, /*AtEndOfTU=*/false);
}

} // end namespace sema
} // end namespace clang
