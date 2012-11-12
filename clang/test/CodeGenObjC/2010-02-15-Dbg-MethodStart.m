// RUN: %clang_cc1 -x objective-c -emit-llvm -g < %s | grep  subprogram | grep "i32 9"
// Test to check that subprogram start location.

@interface Foo
-(int) barMethod;
@end

@implementation Foo
-(int) barMethod {
  int i = 0;
  int j = 1;
  int k = 1;
  return i + j + k;
}
@end
