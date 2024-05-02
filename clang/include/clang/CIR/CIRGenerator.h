//===- CIRGenerator.h - CIR Generation from Clang AST ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform CIR generation from Clang
// AST
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIRGENERATOR_H_
#define CLANG_CIRGENERATOR_H_

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"

namespace cir {
class CIRGenerator : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
};

} // namespace cir

#endif // CLANG_CIRGENERATOR_H_
