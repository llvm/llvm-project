// RUN: %clang_cc1 -std=c++11 -fdebugger-support -fdebugger-cast-result-to-id -funknown-anytype -fsyntax-only -verify %s

extern __unknown_anytype test0a;
extern __unknown_anytype test1a();
extern __unknown_anytype test0b;
extern __unknown_anytype test1b();
extern __unknown_anytype test0c;
extern __unknown_anytype test1c();
extern __unknown_anytype test0d;
extern __unknown_anytype test1d();
extern __unknown_anytype test0d;
extern __unknown_anytype test1d();

@interface A
@end

namespace rdar9416370 {
  void test(id x) {
    if ([x foo]) {} // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
    [x foo];
  }
}

@class NSString; // expected-note {{forward declaration of class here}}
namespace rdar10988847 {
  void test() {
    id s = [NSString stringWithUTF8String:"foo"]; // expected-warning {{receiver 'NSString' is a forward class and corresponding @interface may not exist}}
  }
}

namespace rdar13338107 {
  void test() {
    id x1 = test0a;
    id x2 = test1a();
    A *x3 = test0b;
    A *x4 = test1b();
    auto x5 = test0c;
    auto x6 = test1c();
  }
}
