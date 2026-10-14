// REQUIRES: system-linux,bolt-runtime

// RUN: %clangxx --target=aarch64-unknown-linux-gnu \
// RUN: -mbranch-protection=pac-ret -Wl,-q %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt.exe
// RUN: %t.bolt.exe | FileCheck %s

// CHECK: Exception caught: Exception from bar().

#include <cstdio>
#include <stdexcept>

void bar() { throw std::runtime_error("Exception from bar()."); }

void foo() {
  try {
    bar();
  } catch (const std::exception &e) {
    printf("Exception caught: %s\n", e.what());
  }
}

int main() {
  foo();
  return 0;
}
