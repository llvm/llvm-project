// RUN: %clang_cc1 -emit-llvm %s -o %t
namespace foo {

// RUN: not grep "@a = global i32" %t
extern "C" int a;

// RUN: not grep "@_ZN3foo1bE = global i32" %t
extern int b;

// RUN: grep "@_ZN3foo1cE = global i32" %t | count 1
int c = 5;

// RUN: not grep "@_ZN3foo1dE" %t
extern "C" struct d;

}
