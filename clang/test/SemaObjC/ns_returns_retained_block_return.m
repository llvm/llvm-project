// RUN: %clang_cc1  -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1  -fblocks -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

typedef void (^BT) (void);

BT foo(void)  __attribute__((ns_returns_retained));

@interface I
BT foo(void)  __attribute__((ns_returns_retained));
- (BT) Meth __attribute__((ns_returns_retained));
+ (BT) ClsMeth __attribute__((ns_returns_retained));
@end

@implementation I
BT foo(void)  __attribute__((ns_returns_retained)) {return ^{}; }
- (BT) Meth {return ^{}; }
+ (BT) ClsMeth {return ^{}; }
@end
