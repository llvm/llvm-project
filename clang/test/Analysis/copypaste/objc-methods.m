// RUN: %clang_analyze_cc1 -Wno-objc-root-class -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if we search for clones in Objective-C methods.

@interface A
- (int) setOk : (int) a : (int) b;
@end

@implementation A
- (int) setOk : (int) a : (int) b {  // expected-warning{{Duplicate code detected}}
  if (a > b)
    return a;
  return b;
}
@end

@interface B
- (int) setOk : (int) a : (int) b;
@end

@implementation B
- (int) setOk : (int) a : (int) b { // expected-note{{Similar code here}}
  if (a > b)
    return a;
  return b;
}
@end
