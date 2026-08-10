// RUN: %clang_cc1 -fcxx-exceptions -fexceptions  -triple x86_64-apple-darwin11 -fsyntax-only -verify %s 

@interface NSException @end

namespace test0 {
  void test() {
    try {
    } catch (NSException e) { // expected-error {{cannot catch an Objective-C object by value}}
    }
  }
}
