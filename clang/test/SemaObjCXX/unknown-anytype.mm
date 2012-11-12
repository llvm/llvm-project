// RUN: %clang_cc1 -fdebugger-support -funknown-anytype -fsyntax-only -verify %s

// rdar://problem/9416370
namespace test0 {
  void test(id x) {
    if ([x foo]) {} // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
    [x foo]; // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
  }
}
