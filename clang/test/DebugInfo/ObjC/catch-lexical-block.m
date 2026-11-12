// RUN: %clang_cc1 -debug-info-kind=limited -fobjc-exceptions -emit-llvm %s -o - | FileCheck %s
@interface Foo @end
void f0(void) {
  @try {
    @throw @"a";
  } @catch(Foo *e) {
  }
}

// We should have 3 lexical blocks here at the moment, including one
// for the catch block.
// CHECK: !DILexicalBlock(
// CHECK: !DILocalVariable(
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(
