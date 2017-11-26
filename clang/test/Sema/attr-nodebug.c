// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((nodebug));

void b(int p __attribute__((nodebug))) { // expected-warning {{'nodebug' attribute only applies to functions, function pointers, Objective-C methods, and variables and functions}}
  int b __attribute__((nodebug));
}

void t1() __attribute__((nodebug));

void t2() __attribute__((nodebug(2))); // expected-error {{'nodebug' attribute takes no arguments}}
