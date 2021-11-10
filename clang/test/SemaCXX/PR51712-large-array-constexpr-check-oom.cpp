// REQUIRES: shell
// UNSUPPORTED: win32
// RUN: ulimit -v 1048576
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// This used to require too much memory and crash with OOM.
struct {
  int a, b, c, d;
} arr[1<<30];

