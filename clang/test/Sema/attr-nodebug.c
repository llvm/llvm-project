// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((nodebug));

void b() {
  int b __attribute__((nodebug)); // expected-warning {{'nodebug' only applies to variables with static storage duration and functions}}
} 

void t1() __attribute__((nodebug));

void t2() __attribute__((nodebug(2))); // expected-error {{attribute takes no arguments}}
