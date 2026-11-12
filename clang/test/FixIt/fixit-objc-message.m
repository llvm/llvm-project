// Objective-C recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -Wall -fixit -x objective-c %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror -x objective-c %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -Wall -fixit -x objective-c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror -x objective-c++ %t

@interface A
- (int)method1:(int)x second:(float)y;
+ (int)method2:(int)x second:(double)y;
- (int)getBlah;
@end

void f(A *a, int i, int j) {
  a method1:5+2 second:+(3.14159)];
  a method1:[a method1:3 second:j] second:i++]
  a getBlah];

  int array[17];
  (void)array[a method1:5+2 second:+(3.14159)]];
  (A method2:5+2 second:3.14159]);
  A method2:5+2 second:3.14159]
  if (A method2:5+2 second:3.14159]) { }
}

@interface B : A
- (int)method1:(int)x second:(float)y;
@end

@implementation B
- (int)method1:(int)x second:(float)y {
  super method1:x second:y];
  return super getBlah];
}
@end
