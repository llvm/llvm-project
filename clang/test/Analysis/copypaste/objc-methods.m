// RUN: %clang_cc1 -analyze -Wno-objc-root-class -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if we search for clones in Objective-C methods.

@interface A
- (int) setOk : (int) a : (int) b;
@end

@implementation A
- (int) setOk : (int) a : (int) b {  // expected-warning{{Detected code clone.}}
  if (a > b)
    return a;
  return b;
}
@end

@interface B
- (int) setOk : (int) a : (int) b;
@end

@implementation B
- (int) setOk : (int) a : (int) b { // expected-note{{Related code clone is here.}}
  if (a > b)
    return a;
  return b;
}
@end
