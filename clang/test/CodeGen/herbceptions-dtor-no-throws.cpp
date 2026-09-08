// RUN: not %clang -std=c++20 -fherbceptions -fno-exceptions -S -emit-llvm %s 2>&1 | FileCheck %s

// Destructors must NOT have throws - this should be a compile error.

namespace std {
struct error { void *d; __SIZE_TYPE__ c; };
}

struct Bad {
  // Destructor with throws should be rejected
  // CHECK: error: destructor cannot be declared with a herbception
  ~Bad() throws {}
};
