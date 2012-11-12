// PR1013
// Check to make sure debug symbols use the correct name for globals and
// functions.  Will not assemble if it fails to.
// RUN: %clang_cc1 -emit-llvm -g -o - %s | FileCheck %s

// CHECK: f\01oo"
int foo __asm__("f\001oo");

int bar() {
  return foo;
}
