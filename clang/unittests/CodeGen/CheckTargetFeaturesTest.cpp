#include "../lib/Basic/BuiltinTargetFeatures.h"
#include "gtest/gtest.h"

using namespace llvm;

// These tests are to Check whether CodeGen::TargetFeatures works correctly.
TEST(CheckTargetFeaturesTest, checkBuiltinFeatures) {
  auto doCheck = [](StringRef BuiltinFeatures, StringRef FuncFeatures) {
    SmallVector<StringRef, 1> Features;
    FuncFeatures.split(Features, ',');
    StringMap<bool> SM;
    for (StringRef F : Features)
      SM.insert(std::make_pair(F, true));
    clang::Builtin::TargetFeatures TF(SM);
    return TF.hasRequiredFeatures(BuiltinFeatures);
  };
  // Make sure the basic function ',' and '|' works correctly
  ASSERT_FALSE(doCheck("A,B,C,D", "A"));
  ASSERT_TRUE(doCheck("A,B,C,D", "A,B,C,D"));
  ASSERT_TRUE(doCheck("A|B", "A"));
  ASSERT_FALSE(doCheck("A|B", "C"));

  // Make sure the ',' has higher priority.
  ASSERT_TRUE(doCheck("A|B,C|D", "A"));

  // Make sure the parentheses do change the priority of '|'.
  ASSERT_FALSE(doCheck("(A|B),(C|D)", "A"));
  ASSERT_TRUE(doCheck("(A|B),(C|D)", "A,C"));

  // Make sure the combination in parentheses works correctly.
  ASSERT_FALSE(doCheck("(A,B|C),D", "A,C"));
  ASSERT_FALSE(doCheck("(A,B|C),D", "A,D"));
  ASSERT_TRUE(doCheck("(A,B|C),D", "C,D"));
  ASSERT_TRUE(doCheck("(A,B|C),D", "A,B,D"));

  // Make sure nested parentheses works correctly.
  ASSERT_FALSE(doCheck("(A,(B|C)),D", "C,D"));
  ASSERT_TRUE(doCheck("(A,(B|C)),D", "A,C,D"));
}
