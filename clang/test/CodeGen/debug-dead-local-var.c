// RUN: %clang_cc1 -mllvm -asm-verbose -S -O2 -g %s -o - | FileCheck %s
// Radar 8122864

// Code is not generated for function foo, but preserve type information of
// local variable xyz.
static void foo() {
// CHECK: DW_TAG_structure_type 
  struct X { int a; int b; } xyz;
}

int bar() {
  foo();
  return 1;
}
