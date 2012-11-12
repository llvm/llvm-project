// RUN: %clang_cc1 %s -verify -fsyntax-only
// Note: most of the 'nodebug' tests are in attr-nodebug.c.

// expected-no-diagnostics
class c {
  void t3() __attribute__((nodebug));
};
