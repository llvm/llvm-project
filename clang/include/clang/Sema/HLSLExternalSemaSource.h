//===--- HLSLExternalSemaSource.h - HLSL Sema Source ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HLSLExternalSemaSource interface.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H
#define CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H

#include "clang/Sema/ExternalSemaSource.h"

namespace clang {
class NamespaceDecl;
class Sema;

class HLSLExternalSemaSource : public ExternalSemaSource {
  static char ID;

  Sema *SemaPtr = nullptr;
  NamespaceDecl *HLSLNamespace;

  void defineHLSLVectorAlias();

public:
  ~HLSLExternalSemaSource() override;

  /// Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  void InitializeSema(Sema &S) override;

  /// Inform the semantic consumer that Sema is no longer available.
  void ForgetSema() override { SemaPtr = nullptr; }
};

} // namespace clang

#endif // CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H
