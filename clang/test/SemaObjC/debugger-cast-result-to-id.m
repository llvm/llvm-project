// RUN: %clang_cc1 -funknown-anytype -fsyntax-only -fdebugger-support -fdebugger-cast-result-to-id -verify %s

extern __unknown_anytype test0;
extern __unknown_anytype test1(void);

void test_unknown_anytype_receiver(void) {
  (void)(int)[[test0 unknownMethod] otherUnknownMethod];;
  (void)(id)[[test1() unknownMethod] otherUnknownMethod];
  id x = test0;
  id y = test1();
}

// rdar://10988847
@class NSString; // expected-note {{forward declaration of class here}}

void rdar10988847(void) {
  id s = [NSString stringWithUTF8String:"foo"]; // expected-warning {{receiver 'NSString' is a forward class and corresponding @interface may not exist}}
}
