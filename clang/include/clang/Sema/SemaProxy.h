//=== SemaProxy.h - Sema proxy for effectual constant evaluation ----------===//
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

#ifndef LLVM_CLANG_SEMA_SEMAPROXY_H
#define LLVM_CLANG_SEMA_SEMAPROXY_H

#include "clang/AST/SemaProxy.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class FunctionDecl;
class Sema;

namespace sema {

class EvalProxy : public clang::SemaProxy {
public:
  explicit EvalProxy(clang::Sema &SemaRef) : SemaRef(SemaRef) {}

  void instantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                     FunctionDecl *Function) override;

private:
  Sema &SemaRef;
};

} // end namespace sema
} // end namespace clang

#endif
