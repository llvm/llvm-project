// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

// Make sure pragma pack works inside ObjC methods.
@interface X
@end
@implementation X
- (void)Y {
#pragma pack(push, 1)
  struct x {
    char a;
    int b;
  };
#pragma pack(pop)
  typedef char check_[sizeof (struct x) == 5 ? 1 : -1];
}
@end
