//===- unittests/CodeGen/BufferSourceTest.cpp - MemoryBuffer source tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestCompiler.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

TEST(BufferSourceTest, EmitCXXGlobalInitFunc) {
  // Emitting constructors for global objects involves looking
  // at the source file name. This makes sure that we don't crash
  // if the source file is a memory buffer.
  const char TestProgram[] =
    "class EmitCXXGlobalInitFunc    "
    "{                              "
    "public:                        "
    "   EmitCXXGlobalInitFunc() {}  "
    "};                             "
    "EmitCXXGlobalInitFunc test;    ";

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  TestCompiler Compiler(LO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
}

} // end anonymous namespace
