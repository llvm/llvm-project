// Only run this test where ulimit is known to work well.
// (There's nothing really platform-specific being tested, this is just ulimit).
//
// REQUIRES: system-linux
// UNSUPPORTED: msan
// UNSUPPORTED: asan
// UNSUPPORTED: hwasan
//
// RUN: ulimit -v 1048576
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -triple=x86_64 %s
// expected-no-diagnostics

// This used to require too much memory and crash with OOM.
struct {
  int a, b, c, d;
} arr[1<<30];

