// RUN: %clang_cc1 %s -verify

@interface NSWidget @end

__attribute__((objc_externally_retained)) void f(NSWidget *p) { // expected-warning{{'objc_externally_retained' attribute ignored}}
  __attribute__((objc_externally_retained)) NSWidget *w; // expected-warning{{'objc_externally_retained' attribute ignored}}
}
