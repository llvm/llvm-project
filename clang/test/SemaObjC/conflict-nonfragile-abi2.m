// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://8225011

int glob;

@interface I
@property int glob;
@property int p;
@property int le;
@property int l;
@property int ls;
@property int r;
@end

// rdar://9027673
// Warning on future name lookup rule is removed.
@implementation I
- (int) Meth { return glob; } // no warning
@synthesize glob;
// rdar://8248681
- (int) Meth1: (int) p {
  extern int le;
  int l = 1;
  static int ls;
  register int r;
  p = le + ls + r;
  return l;
}
@dynamic p;
@dynamic le;
@dynamic l;
@dynamic ls;
@dynamic r;
@end


