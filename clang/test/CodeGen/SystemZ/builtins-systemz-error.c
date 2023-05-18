// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu zEC12 -triple s390x-unknown-unknown \
// RUN: -Wall -Wno-unused -Werror -fsyntax-only -verify %s

void test1(void) {
  __builtin_tabort (0);   // expected-error {{invalid transaction abort code}}
  __builtin_tabort (255); // expected-error {{invalid transaction abort code}}
}

