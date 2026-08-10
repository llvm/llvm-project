//=== unittests/CodeGen/IncrementalProcessingTest.cpp - IncrementalCodeGen ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

#include <memory>

#if defined(_AIX) || defined(__MVS__)
#define CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
#endif

using namespace llvm;
using namespace clang;

namespace {

class IncrementalProcessingTest : public InterpreterTestBase {};

// Incremental processing produces several modules, all using the same "main
// file". Make sure CodeGen can cope with that, e.g. for static initializers.
const char TestProgram1[] = "extern \"C\" int funcForProg1() { return 17; }\n"
                            "struct EmitCXXGlobalInitFunc1 {\n"
                            "   EmitCXXGlobalInitFunc1() {}\n"
                            "} test1;";

const char TestProgram2[] = "extern \"C\" int funcForProg2() { return 42; }\n"
                            "struct EmitCXXGlobalInitFunc2 {\n"
                            "   EmitCXXGlobalInitFunc2() {}\n"
                            "} test2;";

const Function *getGlobalInit(llvm::Module *M) {
  for (const auto &Func : *M)
    if (Func.hasName() && Func.getName().starts_with("_GLOBAL__sub_I_"))
      return &Func;

  return nullptr;
}

TEST_F(IncrementalProcessingTest, EmitCXXGlobalInitFunc) {
  std::vector<const char *> ClangArgv = {"-Xclang", "-emit-llvm-only"};
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgv);
  auto CI = cantFail(CB.CreateCpp());
  auto Interp = llvm::cantFail(Interpreter::create(std::move(CI)));

  std::array<clang::PartialTranslationUnit *, 2> PTUs;

  PTUs[0] = &llvm::cantFail(Interp->Parse(TestProgram1));
  ASSERT_TRUE(PTUs[0]->TheModule);
  ASSERT_TRUE(PTUs[0]->TheModule->getFunction("funcForProg1"));

  PTUs[1] = &llvm::cantFail(Interp->Parse(TestProgram2));
  ASSERT_TRUE(PTUs[1]->TheModule);
  ASSERT_TRUE(PTUs[1]->TheModule->getFunction("funcForProg2"));
  // First code should not end up in second module:
  ASSERT_FALSE(PTUs[1]->TheModule->getFunction("funcForProg1"));

  // Make sure global inits exist and are unique:
  const Function *GlobalInit1 = getGlobalInit(PTUs[0]->TheModule.get());
  ASSERT_TRUE(GlobalInit1);

  const Function *GlobalInit2 = getGlobalInit(PTUs[1]->TheModule.get());
  ASSERT_TRUE(GlobalInit2);

  ASSERT_FALSE(GlobalInit1->getName() == GlobalInit2->getName());
}

} // end anonymous namespace
