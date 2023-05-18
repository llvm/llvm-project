// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

struct S {};

@interface I
  @property (readonly) S* prop __attribute__((os_returns_retained));
  - (S*) generateS __attribute__((os_returns_retained));
  - (void) takeS:(S*) __attribute__((os_consumed)) s;
@end

typedef __attribute__((os_returns_retained)) id (^blockType)(); // expected-warning{{'os_returns_retained' attribute only applies to functions, Objective-C methods, Objective-C properties, and parameters}}

__auto_type b = ^ id (id filter)  __attribute__((os_returns_retained))  { // expected-warning{{'os_returns_retained' attribute only applies to functions, Objective-C methods, Objective-C properties, and parameters}}
  return filter;
};
