#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;

static FunctionDecl *findFunction(ASTContext &Context, llvm::StringRef Name) {
  for (Decl *Declaration : Context.getTranslationUnitDecl()->decls())
    if (auto *Function = dyn_cast<FunctionDecl>(Declaration))
      if (Function->getName() == Name)
        return Function;
  return nullptr;
}

TEST(FunctionFeatureMap, DefaultsAndCopies) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "void first(); void second();", {"-target", "aarch64-none-linux-gnu"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *First = findFunction(Context, "first");
  auto *Second = findFunction(Context, "second");
  ASSERT_NE(First, nullptr);
  ASSERT_NE(Second, nullptr);

  const auto &Features = Context.getFunctionFeatureMap(First);
  EXPECT_EQ(&Features, &Context.getFunctionFeatureMap(Second));
  EXPECT_FALSE(Features.lookup("sve"));
  llvm::StringMap<bool> Copy;
  Context.getFunctionFeatureMap(Copy, First);
  Copy["sve"] = true;
  EXPECT_FALSE(Features.lookup("sve"));

  const FunctionDecl *NoFunction = nullptr;
  llvm::StringMap<bool> LegacyNullFeatures;
  Context.getFunctionFeatureMap(LegacyNullFeatures, NoFunction);
  const auto &NullFeatures = Context.getFunctionFeatureMap(NoFunction);
  EXPECT_EQ(&NullFeatures, &Context.getFunctionFeatureMap(NoFunction));
  EXPECT_EQ(NullFeatures.size(), LegacyNullFeatures.size());
  for (const auto &Entry : LegacyNullFeatures) {
    auto Found = NullFeatures.find(Entry.getKey());
    ASSERT_NE(Found, NullFeatures.end());
    EXPECT_EQ(Found->getValue(), Entry.getValue());
  }
}

TEST(FunctionFeatureMap, TargetAttributeChanges) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "__attribute__((target(\"+sve\"))) void target();",
      {"-target", "aarch64-none-linux-gnu"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *Function = findFunction(Context, "target");
  ASSERT_NE(Function, nullptr);
  const auto &Features = Context.getFunctionFeatureMap(Function);
  EXPECT_TRUE(Features.lookup("sve"));
  EXPECT_EQ(&Features, &Context.getFunctionFeatureMap(Function));

  Function->dropAttr<TargetAttr>();
  EXPECT_FALSE(Context.getFunctionFeatureMap(Function).lookup("sve"));
  EXPECT_TRUE(Features.lookup("sve"));
  Function->addAttr(TargetAttr::CreateImplicit(Context, "+sve"));
  EXPECT_TRUE(Context.getFunctionFeatureMap(Function).lookup("sve"));
}

TEST(FunctionFeatureMap, MultiversionIndices) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "__attribute__((target_clones(\"sve\", \"default\"))) "
      "int target() { return 0; }",
      {"-target", "aarch64-none-linux-gnu", "-rtlib=compiler-rt"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *Function = findFunction(Context, "target");
  ASSERT_NE(Function, nullptr);
  ASSERT_TRUE(Function->hasAttr<TargetClonesAttr>());
  GlobalDecl First = GlobalDecl().getWithDecl(Function);
  GlobalDecl Second = First.getWithMultiVersionIndex(1);
  const auto &FirstFeatures = Context.getFunctionFeatureMap(First);
  const auto &SecondFeatures = Context.getFunctionFeatureMap(Second);
  EXPECT_NE(&FirstFeatures, &SecondFeatures);
  EXPECT_NE(FirstFeatures.lookup("sve"), SecondFeatures.lookup("sve"));
  EXPECT_EQ(&FirstFeatures, &Context.getFunctionFeatureMap(First));
  EXPECT_EQ(&SecondFeatures, &Context.getFunctionFeatureMap(Second));
}

TEST(FunctionFeatureMap, StableReferences) {
  std::string Code;
  for (unsigned Index = 0; Index != 100; ++Index)
    Code += "__attribute__((target(\"+sve\"))) void target" +
            std::to_string(Index) + "();";
  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-target", "aarch64-none-linux-gnu"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *First = findFunction(Context, "target0");
  ASSERT_NE(First, nullptr);
  const auto &Features = Context.getFunctionFeatureMap(First);
  for (Decl *Declaration : Context.getTranslationUnitDecl()->decls())
    if (auto *Function = dyn_cast<FunctionDecl>(Declaration))
      Context.getFunctionFeatureMap(Function);
  EXPECT_EQ(&Features, &Context.getFunctionFeatureMap(First));
  EXPECT_TRUE(Features.lookup("sve"));
}
