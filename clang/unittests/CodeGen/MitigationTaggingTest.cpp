#include "TestCompiler.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/MitigationTagging.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MitigationMarker.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::CodeGen;

namespace {
static const char TestProgram[] = "void SomeTestFunction(int x, int y)  "
                                  "{                                    "
                                  "   char buf[x];                      "
                                  "   while(y)                          "
                                  "      buf[y--] = 0;                  "
                                  "}                                    ";
static constexpr char TestFunctionMangledName[] = "_Z16SomeTestFunctionii";

static int GetEnablementForMitigation(llvm::Function *FuncPtr,
                                      StringRef MitigationStr) {
  auto *MD = FuncPtr->getMetadata(MitigationStr);
  // If not found, it just means the mitigation is likely not available
  if (!MD)
    return 0;

  if (MD->getNumOperands() != 1)
    return -1;

  auto *ConstAsMeta = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!ConstAsMeta)
    return -2;

  auto *Value = ConstAsMeta->getValue();

  return Value->isOneValue();
}

TEST(MitigationTaggingTest, FlagDisabled) {
  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;

  clang::CodeGenOptions CGO;

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  auto Mapping = GetMitigationMetadataMapping();
  auto *MD =
      FuncPtr->getMetadata(Mapping[MitigationKey::STACK_CLASH_PROTECTION]);
  ASSERT_TRUE(MD == nullptr);
}

TEST(MitigationTaggingTest, MetadataEnabledOnly) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that all mitigations disabled
  for (const auto &[MitigationKey, MitigationStr] :
       GetMitigationMetadataMapping()) {
    EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
}

TEST(MitigationTaggingTest, AutoVarInitZeroEnabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  LO.setTrivialAutoVarInit(LangOptions::TrivialAutoVarInitKind::Zero);

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::AUTO_VAR_INIT)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 1);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(mitigationMetadataMapping.find(MitigationKey::AUTO_VAR_INIT),
            mitigationMetadataMapping.end());
}

TEST(MitigationTaggingTest, StackClashEnabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;
  CGO.StackClashProtector = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::STACK_CLASH_PROTECTION)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 1);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(
      mitigationMetadataMapping.find(MitigationKey::STACK_CLASH_PROTECTION),
      mitigationMetadataMapping.end());
}

TEST(MitigationTaggingTest, StackProtectorDisabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  LO.setStackProtector(LangOptions::SSPOff);

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::STACK_PROTECTOR)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(mitigationMetadataMapping.find(MitigationKey::STACK_PROTECTOR),
            mitigationMetadataMapping.end());
}

TEST(MitigationTaggingTest, StackProtectorEnabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  LO.setStackProtector(LangOptions::SSPOn);

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::STACK_PROTECTOR)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 1);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(mitigationMetadataMapping.find(MitigationKey::STACK_PROTECTOR),
            mitigationMetadataMapping.end());
}

TEST(MitigationTaggingTest, StackProtectorStrongEnabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  LO.setStackProtector(LangOptions::SSPStrong);

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::STACK_PROTECTOR_STRONG)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 1);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(
      mitigationMetadataMapping.find(MitigationKey::STACK_PROTECTOR_STRONG),
      mitigationMetadataMapping.end());
}

TEST(MitigationTaggingTest, StackProtectorAllEnabled) {
  clang::CodeGenOptions CGO;
  CGO.MitigationAnalysis = true;

  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  LO.CPlusPlus11 = 1;
  LO.setStackProtector(LangOptions::SSPReq);

  TestCompiler Compiler(LO, CGO);
  Compiler.init(TestProgram);

  clang::ParseAST(Compiler.compiler.getSema(), false, false);
  auto M =
      static_cast<clang::CodeGenerator &>(Compiler.compiler.getASTConsumer())
          .GetModule();
  auto FuncPtr = M->getFunction(TestFunctionMangledName);
  ASSERT_TRUE(FuncPtr != nullptr);

  // Check that the correct mitigations are enabled
  auto mitigationMetadataMapping = GetMitigationMetadataMapping();
  for (const auto &[MitigationKey, MitigationStr] : mitigationMetadataMapping) {
    if (MitigationKey == MitigationKey::STACK_PROTECTOR_ALL)
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 1);
    else
      EXPECT_EQ(GetEnablementForMitigation(FuncPtr, MitigationStr), 0);
  }
  EXPECT_NE(mitigationMetadataMapping.find(MitigationKey::STACK_PROTECTOR_ALL),
            mitigationMetadataMapping.end());
}

} // end anonymous namespace
