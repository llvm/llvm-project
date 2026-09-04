// RUN: %clangxx_tysan -O0 %s -o %t && %run %t 2>&1 | FileCheck --implicit-check-not ERROR %s

// https://github.com/llvm/llvm-project/issues/208651

#include <stdio.h>

class A {
public:
  virtual ~A() = default;
  char byte;
};

static A a;

int main() {
  printf("done\n");
  // CHECK: done
  return 0;
}
