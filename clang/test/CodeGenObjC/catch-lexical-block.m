// RUN: %clang_cc1 -g -fobjc-exceptions -emit-llvm %s -o - | FileCheck %s
@interface Foo @end
void f0() {
  @try {
    @throw @"a";
  } @catch(Foo *e) {
  }
}

// We should have 4 lexical blocks here at the moment, including one
// for the catch block.
// CHECK: lexical_block
// CHECK: lexical_block
// CHECK: lexical_block
// CHECK: auto_variable
// CHECK: lexical_block
