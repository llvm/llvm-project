// RUN: %clang_cpp_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_cpp_csi_toolc %tooldir/function-call-count-tool.c -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_cpp_csi_c %s -o %t.o
// RUN: %clang_cpp_csi %t.o %t-tool.o %csirtlib -o %t
// RUN: %run %t | FileCheck %s

// In this test, the TIX uses the STL, but the tool does not.

#include <stdio.h>
#include <vector>

int main(int argc, char **argv) {
  printf("One call.\n");
  printf("Two calls.\n");
  std::vector<int> stackVec;
  for (int i = 0; i < 10; i++)
    stackVec.push_back(i);
  // CHECK: num_function_calls =
  // CHECK-NOT: {{0|1}}
  return 0;
}
