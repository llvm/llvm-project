//=== unittests/CodeGen/ReloadLinkModulesTest.cpp - reloadLinkModules test ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests CodeGenAction::reloadLinkModules. BackendConsumer::LinkInModules()
// consumes the -mlink-builtin-bitcode modules (it clears the list after
// linking), so a BackendConsumer that is reused across translation units --
// as in clang-repl's incremental compilation -- would link the bitcode into
// the first module only. reloadLinkModules() reloads and reseeds those modules
// so each subsequent translation unit links them again.
//
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/BackendConsumer.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;

namespace {

// Writes a bitcode module defining `i32 @linked_fn()` to a temporary file and
// returns its path. The temporary is registered for removal by the caller.
static std::string writeLinkBitcodeFile(SmallVectorImpl<char> &PathStorage) {
  LLVMContext Ctx;
  auto M = std::make_unique<llvm::Module>("linked", Ctx);

  auto *FT =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(Ctx), /*isVarArg=*/false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "linked_fn", *M);
  auto *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> B(BB);
  B.CreateRet(ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 42));

  int FD = -1;
  std::error_code EC =
      sys::fs::createTemporaryFile("reload-link", "bc", FD, PathStorage);
  EXPECT_FALSE(EC) << EC.message();
  {
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    WriteBitcodeToFile(*M, OS);
  }
  return std::string(PathStorage.data(), PathStorage.size());
}

// Returns true if the module contains a (non-declaration) definition of
// `linked_fn`, i.e. the bitcode library was linked in.
static bool hasLinkedFn(llvm::Module &M) {
  Function *F = M.getFunction("linked_fn");
  return F && !F->isDeclaration();
}

TEST(ReloadLinkModulesTest, ReloadReseedsConsumedLinkModules) {
  // A temporary bitcode file to be linked in via -mlink-builtin-bitcode.
  SmallString<128> BCPath;
  std::string BCFile = writeLinkBitcodeFile(BCPath);
  FileRemover BCRemover(BCPath);

  // The LLVM context is owned by the test so that the scratch modules below
  // live in the same context as the modules the action loads.
  LLVMContext Context;

  CompilerInstance Compiler;
  Compiler.getCodeGenOpts().LinkBitcodeFiles.push_back(
      {BCFile, /*PropagateAttrs=*/false, /*Internalize=*/false,
       /*LinkFlags=*/0});

  Compiler.setVirtualFileSystem(vfs::getRealFileSystem());
  Compiler.createDiagnostics();

  Triple HostTriple(Triple::normalize(sys::getProcessTriple()));
  Compiler.getTargetOpts().Triple = HostTriple.getTriple();
  Compiler.setTarget(TargetInfo::CreateTargetInfo(Compiler.getDiagnostics(),
                                                  Compiler.getTargetOpts()));
  ASSERT_TRUE(Compiler.hasTarget());

  // EmitLLVMOnlyAction builds the module in memory and emits no output file,
  // which is what clang-repl uses.
  EmitLLVMOnlyAction Action(&Context);

  auto Buffer = MemoryBuffer::getMemBuffer("", "reload-link-input.c");
  FrontendInputFile Input(Buffer->getMemBufferRef(), InputKind(Language::C));

  // BeginSourceFile creates the BackendConsumer, which loads the link modules.
  // It does not parse (that happens in Execute), which is all we need.
  ASSERT_TRUE(Action.BeginSourceFile(Compiler, Input));
  ASSERT_NE(Action.BEConsumer, nullptr);

  // First translation unit: the link module is present and gets linked in.
  auto M1 = std::make_unique<llvm::Module>("tu1", Context);
  ASSERT_FALSE(Action.BEConsumer->LinkInModules(M1.get()));
  EXPECT_TRUE(hasLinkedFn(*M1)) << "first TU should link in the bitcode";

  // Second translation unit without reloading: LinkInModules consumed the list,
  // so nothing is linked in. This is the bug reloadLinkModules exists to fix.
  auto M2 = std::make_unique<llvm::Module>("tu2", Context);
  ASSERT_FALSE(Action.BEConsumer->LinkInModules(M2.get()));
  EXPECT_FALSE(hasLinkedFn(*M2))
      << "link modules should have been consumed by the first TU";

  // After reloading, the link module is reseeded and links in again.
  Action.reloadLinkModules(Compiler);
  auto M3 = std::make_unique<llvm::Module>("tu3", Context);
  ASSERT_FALSE(Action.BEConsumer->LinkInModules(M3.get()));
  EXPECT_TRUE(hasLinkedFn(*M3))
      << "reloadLinkModules should reseed the link modules";

  Action.EndSourceFile();
}

} // namespace
