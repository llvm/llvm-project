// RUN: %clang_cc1 %s -verify 

void foo() {
  // expected-error@+1{{an attribute list cannot appear here}}
__attribute__((aligned(64)))
#pragma clang attribute push(__attribute__((uninitialized)), apply_to = any(variable(is_local)))
{
  int f;
}
#pragma clang attribute pop
}

void foo2() {
  // expected-error@+1{{an attribute list cannot appear here}}
__attribute__((aligned(64)))
#pragma clang __debug dump foo
}

void foo3() {
  // expected-error@+1{{an attribute list cannot appear here}}
  [[nodiscard]]
#pragma clang attribute push(__attribute__((uninitialized)), apply_to = any(variable(is_local)))
{
  int f;
}
#pragma clang attribute pop
}

void foo4() {
  // expected-error@+1{{an attribute list cannot appear here}}
  [[nodiscard]]
#pragma clang __debug dump foo
}
