// RUN: %clang_cc1 %s -emit-llvm -o - | opt -std-compile-opts | llc | \
// RUN:    not grep _foo2

void foo() __asm__("foo2");

void bar() {
  foo();
}
