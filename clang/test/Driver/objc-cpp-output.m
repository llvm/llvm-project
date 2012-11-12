// RUN: %clang -x objc-cpp-output -c %s -o /dev/null

// PR13820
// REQUIRES: LP64

// Should compile without errors
@protocol P
- (void)m;
@end
void f() {}
