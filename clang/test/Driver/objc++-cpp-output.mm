// RUN: %clang -fexceptions -emit-llvm -x objc++-cpp-output -S %s -o /dev/null
// RUN: %clang -fexceptions -emit-llvm -x objc++-cpp-output -S %s -o /dev/null -### 2>&1 | FileCheck %s

// PR13820
// REQUIRES: LP64

// Should compile without errors
@protocol P
- (void)m;
@end
void f() {}
class C {};

// Make sure the driver is passing all the necessary exception flags.
// CHECK: "-fobjc-exceptions"
// CHECK: "-fcxx-exceptions"
// CHECK: "-fexceptions" 
