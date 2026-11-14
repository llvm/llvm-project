// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x objective-c -O2 \
// RUN:   -disable-llvm-passes -fopenmp -ffast-math -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// The method has no FunctionDecl to take the target features from, so the
// mapping is built from the TU features.
#pragma omp declare simd notinbranch
double sin(double);

@interface A
- (double)m:(double)x;
@end
@implementation A
- (double)m:(double)x {
  return sin(x);
}
@end

// CHECK: call{{.*}} double @sin({{.*}}) #[[MAPPED:[0-9]+]]
// CHECK-DAG: attributes #[[MAPPED]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin"
