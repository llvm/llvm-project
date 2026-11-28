#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

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
  EXPECT_EQ(&Features, &Context.getFunctionFeatureMap(NoFunction));
}

TEST(FunctionFeatureMap, SharedAttributeContents) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "__attribute__((target(\"+sve\"))) void first();"
      "__attribute__((target(\"+sve\"))) void second();",
      {"-target", "aarch64-none-linux-gnu"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *First = findFunction(Context, "first");
  auto *Second = findFunction(Context, "second");
  ASSERT_NE(First, nullptr);
  ASSERT_NE(Second, nullptr);
  ASSERT_NE(First->getAttr<TargetAttr>(), Second->getAttr<TargetAttr>());
  const auto &Features = Context.getFunctionFeatureMap(First);
  EXPECT_TRUE(Features.lookup("sve"));
  EXPECT_EQ(&Features, &Context.getFunctionFeatureMap(Second));
}

TEST(FunctionFeatureMap, MultiversionIndices) {
  auto AST = tooling::buildASTFromCodeWithArgs(
      "__attribute__((target_clones(\"sve\", \"default\"))) "
      "int target() { return 0; }"
      "__attribute__((target_clones(\"sve\", \"default\"))) "
      "int peer() { return 0; }",
      {"-target", "aarch64-none-linux-gnu", "-rtlib=compiler-rt"});
  ASSERT_TRUE(AST);
  ASTContext &Context = AST->getASTContext();
  auto *Function = findFunction(Context, "target");
  auto *Peer = findFunction(Context, "peer");
  ASSERT_NE(Function, nullptr);
  ASSERT_NE(Peer, nullptr);
  ASSERT_TRUE(Function->hasAttr<TargetClonesAttr>());
  GlobalDecl First = GlobalDecl().getWithDecl(Function);
  GlobalDecl Second = First.getWithMultiVersionIndex(1);
  const auto &FirstFeatures = Context.getFunctionFeatureMap(First);
  const auto &SecondFeatures = Context.getFunctionFeatureMap(Second);
  EXPECT_NE(&FirstFeatures, &SecondFeatures);
  EXPECT_NE(FirstFeatures.lookup("sve"), SecondFeatures.lookup("sve"));
  EXPECT_EQ(&FirstFeatures, &Context.getFunctionFeatureMap(First));
  EXPECT_EQ(&SecondFeatures, &Context.getFunctionFeatureMap(Second));
  GlobalDecl PeerFirst = GlobalDecl().getWithDecl(Peer);
  EXPECT_EQ(&FirstFeatures, &Context.getFunctionFeatureMap(PeerFirst));
  EXPECT_EQ(&SecondFeatures, &Context.getFunctionFeatureMap(
                                 PeerFirst.getWithMultiVersionIndex(1)));
}
