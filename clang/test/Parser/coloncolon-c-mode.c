// Regression test for GH-208044: clang should not hang when parsing
// `::` in C mode.
// RUN: %clang_cc1 -x c -std=c23 -fsyntax-only %s
int sink = (::i);
