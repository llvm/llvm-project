// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - | FileCheck %s --implicit-check-not=should_not_be_used

void should_be_used_1();
void should_be_used_2();
void should_be_used_3();
void should_not_be_used();

struct A {
  constexpr explicit operator bool() const {
    return true;
  }
};

void f() {
  if constexpr (false)
    should_not_be_used();
  else
    should_be_used_1();

  if constexpr (true || ({ label: false; }))
    should_be_used_2();
  else {
    goto foo;
foo: should_not_be_used();
  }
  if constexpr (A())
    should_be_used_3();
  else
    should_not_be_used();
}

// CHECK: should_be_used_1
// CHECK: should_be_used_2
// CHECK: should_be_used_3
