#include "TestCompiler.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

TEST(MitigationTaggingTest, FlagDisabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD == nullptr);
}

TEST(MitigationTaggingTest, MetadataEnabledOnly) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_FALSE(MDs["auto-var-init"]);
  EXPECT_FALSE(MDs["stack-clash-protection"]);
  EXPECT_FALSE(MDs["stack-protector"]);
  EXPECT_FALSE(MDs["stack-protector-strong"]);
  EXPECT_FALSE(MDs["stack-protector-all"]);
}

TEST(MitigationTaggingTest, AutoVarInitZeroEnabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_TRUE(MDs["auto-var-init"]);
  EXPECT_FALSE(MDs["stack-clash-protection"]);
  EXPECT_FALSE(MDs["stack-protector"]);
  EXPECT_FALSE(MDs["stack-protector-strong"]);
  EXPECT_FALSE(MDs["stack-protector-all"]);
}

TEST(MitigationTaggingTest, StackClashEnabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_FALSE(MDs["auto-var-init"]);
  EXPECT_TRUE(MDs["stack-clash-protection"]);
  EXPECT_FALSE(MDs["stack-protector"]);
  EXPECT_FALSE(MDs["stack-protector-strong"]);
  EXPECT_FALSE(MDs["stack-protector-all"]);
}

TEST(MitigationTaggingTest, StackProtectorEnabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_FALSE(MDs["auto-var-init"]);
  EXPECT_FALSE(MDs["stack-clash-protection"]);
  EXPECT_TRUE(MDs["stack-protector"]);
  EXPECT_FALSE(MDs["stack-protector-strong"]);
  EXPECT_FALSE(MDs["stack-protector-all"]);
}

TEST(MitigationTaggingTest, StackProtectorStrongEnabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_FALSE(MDs["auto-var-init"]);
  EXPECT_FALSE(MDs["stack-clash-protection"]);
  EXPECT_TRUE(MDs["stack-protector"]);
  EXPECT_TRUE(MDs["stack-protector-strong"]);
  EXPECT_FALSE(MDs["stack-protector-all"]);
}

TEST(MitigationTaggingTest, StackProtectorAllEnabled) {
  const char TestProgram[] = "void HasStackProtector(int x, int y) "
                             "{                                    "
                             "   char buf[x];                      "
                             "   while(y)                          "
                             "      buf[y--] = 0;                  "
                             "}                                    ";

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
  auto FuncPtr = M->begin();

  auto *MD = FuncPtr->getMetadata("security_mitigations");
  ASSERT_TRUE(MD != nullptr);

  // Get All Enabled Mitigations
  std::unordered_map<std::string, bool> MDs;
  for (unsigned i = 0, n = MD->getNumOperands(); i != n; ++i) {
    if (MD->getOperand(i) == nullptr)
      continue;

    auto *node = dyn_cast<MDNode>(MD->getOperand(i));
    if (node == nullptr)
      continue;

    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getNumOperands(), 2);

    auto *mds = dyn_cast<MDString>(node->getOperand(0));
    auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
    EXPECT_FALSE(!mds || !cam);

    auto *ci = cam->getValue();
    ASSERT_NE(ci, nullptr);

    auto mitigationName = std::string(mds->getString());
    ASSERT_TRUE(MDs.find(mitigationName) == MDs.end());
    MDs[mitigationName] = ci->isOneValue();
  }

  // Check that the correct mitigations are enabled
  EXPECT_FALSE(MDs["auto-var-init"]);
  EXPECT_FALSE(MDs["stack-clash-protection"]);
  EXPECT_TRUE(MDs["stack-protector"]);
  EXPECT_TRUE(MDs["stack-protector-strong"]);
  EXPECT_TRUE(MDs["stack-protector-all"]);
}

} // end anonymous namespace
