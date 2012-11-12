// RUN: %clang_cc1 -fdebugger-support -fdebugger-cast-result-to-id -funknown-anytype -fsyntax-only -verify %s

// rdar://problem/9416370
namespace test0 {
  void test(id x) {
    if ([x foo]) {} // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
    [x foo];
  }
}

// rdar://10988847
@class NSString; // expected-note {{forward declaration of class here}}
namespace test1 {
  void rdar10988847() {
    id s = [NSString stringWithUTF8String:"foo"]; // expected-warning {{receiver 'NSString' is a forward class and corresponding @interface may not exist}}
  }
}
