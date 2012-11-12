// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result
// DISABLE: mingw32

@interface A
- (id)retain;
- (id)autorelease;
- (oneway void)release;
- (void)dealloc;
@end

void test1(A *a) {
  [a dealloc];
}

@interface Test2 : A
- (void) dealloc;
@end

@implementation Test2
- (void) dealloc {
  [super dealloc];
}
@end
