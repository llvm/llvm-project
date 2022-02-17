// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited < %s | FileCheck %s
// Test to check number of lexical scope identified in debug info.
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(

extern int bar(void);
extern void foobar(void);
void foo(int s) {
  unsigned loc = 0;
  if (s) {
    if (bar()) {
      foobar();
    }
  } else {
    loc = 1;
    if (bar()) {
      loc = 2;
    }
  }
}
