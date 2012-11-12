// RUN: %clang  -g -S -fverbose-asm %s -o - | FileCheck %s

// CHECK: TAG_namespace
namespace A {
  enum numbers {
    ZERO,
    ONE
  };
}

using namespace A;
numbers n;
