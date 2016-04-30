// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((nodebug));

void b() {
  int b __attribute__((nodebug)); // expected-warning {{'nodebug' attribute only applies to functions and global variables}}
}

void t1() __attribute__((nodebug));

void t2() __attribute__((nodebug(2))); // expected-error {{'nodebug' attribute takes no arguments}}
