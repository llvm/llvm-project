//===- LLVMTestBase.h - Test fixure for LLVM dialect tests ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test fixure for LLVM dialect tests.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_UNITTEST_DIALECT_LLVMIR_LLVMTESTBASE_H
#define AIIR_UNITTEST_DIALECT_LLVMIR_LLVMTESTBASE_H

#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/AIIRContext.h"
#include "gtest/gtest.h"

class LLVMIRTest : public ::testing::Test {
protected:
  LLVMIRTest() { context.getOrLoadDialect<aiir::LLVM::LLVMDialect>(); }

  aiir::AIIRContext context;
};

#endif
