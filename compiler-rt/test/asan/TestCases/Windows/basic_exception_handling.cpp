// RUN: %clangxx_asan %s -o %t
// RUN: %run %t | FileCheck %s

// This test tests that declaring a parameter in a catch-block does not produce a false positive
// ASan error on Windows.

// This code is based on the repro in https://github.com/google/sanitizers/issues/749
#include <cstdio>
#include <exception>
#include <stdexcept>

void throwInFunction() { throw std::runtime_error("test2"); }

int main() {
  // case 1: direct throw
  try {
    throw std::runtime_error("test1");
  } catch (const std::exception &ex) {
    puts(ex.what());
    // CHECK: test1
  }

  // case 2: throw in function
  try {
    throwInFunction();
  } catch (const std::exception &ex) {
    puts(ex.what());
    // CHECK: test2
  }

  printf("Success!\n");
  // CHECK: Success!
  return 0;
}
