// RUN: %clang_cc1 -std=c++1y -Wno-unused-value -fsyntax-only -verify -fobjc-arc %s

// expected-no-diagnostics
__attribute__((objc_root_class))
@interface NSString
@end

// rdar://problem/22344904
void testResultTypeDeduction(int i) {
  auto x = [i] {
    switch (i) {
    case 0:
      return @"foo";

    default:
      return @"bar";
    }
  };
}
