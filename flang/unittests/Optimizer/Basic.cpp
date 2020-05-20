#include "flang/Optimizer/Support/InternalNames.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace fir;
using namespace llvm;

TEST(genericName, MyTest) {
        NameUniquer obj;
        std::string val = obj.doCommonBlock("hello");
        std::cout  << val;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

