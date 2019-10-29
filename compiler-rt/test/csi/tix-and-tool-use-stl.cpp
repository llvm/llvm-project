// RUN: %clang_cpp_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_cpp_csi_toolc %tooldir/function-call-count-uses-stl-tool.cpp -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_cpp_csi_c %s -o %t.o
// RUN: %clang_cpp_csi %t.o %t-tool.o -o %t
// RUN: %run %t | FileCheck %s

// In this test, both the TIX and the tool use the STL.  The resulting
// function call count should be the same as the test whose tool does
// not use the STL, because we want the tool's usage of STL not to be
// instrumented.

#include <stdio.h>
#include <vector>

int main(int argc, char **argv) {
  printf("One call.\n");
  printf("Two calls.\n");
  std::vector<int> stackVec;
  for (int i = 0; i < 10; i++)
    stackVec.push_back(i);
  // CHECK: num_function_calls = 412
  return 0;
}
