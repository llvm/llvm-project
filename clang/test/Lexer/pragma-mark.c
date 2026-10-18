// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsyntax-only -fms-extensions -DTEST_MS_PRAGMA -verify
// RUN: %clang_cc1 %s -E -fms-extensions -DTEST_MS_PRAGMA -verify | FileCheck %s
// expected-no-diagnostics

// Lexer diagnostics shouldn't be included in #pragma mark.
#pragma mark Mike's world
_Pragma("mark foo ' bar")

#define X(S) _Pragma(S)
X("mark foo ' bar")

int i;

#ifdef TEST_MS_PRAGMA
// __pragma has already lexed and expanded its arguments. It must consume the
// captured tokens without requiring a character lexer, including annotations
// introduced by nested pragmas.
__pragma(mark)
__pragma(mark foo) int after_mark;
__pragma(mark foo _Pragma("weak foobar")) int after_nested_mark;
int use_after_marks(void) { return after_mark + after_nested_mark; }
// CHECK: int after_mark;
// CHECK: int after_nested_mark;
// CHECK: int use_after_marks(void) { return after_mark + after_nested_mark; }
#endif
