// RUN: %clang -emit-llvm -x objc-cpp-output -S %s -o /dev/null

// PR13820
// REQUIRES: LP64

// Should compile without errors
@protocol P
- (void)m;
@end
void f() {}
