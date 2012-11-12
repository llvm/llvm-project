// RUN: %clang_cc1 -masm-verbose -S -g %s -o - | FileCheck %s
// Radar 9468526
@interface I {
  int _p1;
}
@property int p1;
@end

@implementation I
@synthesize p1 = _p1;
@end

int main() {
  I *myi;
  myi.p1 = 2;
  return 0;
}

// FIXME: Make this test ir files.
// CHECK:       .loc    2 6 0
