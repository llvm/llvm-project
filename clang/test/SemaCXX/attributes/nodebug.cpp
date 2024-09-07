// RUN: %clang_cc1 %s -std=c++11 -verify -fsyntax-only
// Note: most of the 'nodebug' tests are in attr-nodebug.c.

// expected-no-diagnostics
class c {
  void t3() __attribute__((nodebug));
};

[[gnu::nodebug]] void f() {}
