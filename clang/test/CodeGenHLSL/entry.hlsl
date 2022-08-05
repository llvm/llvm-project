// RUN: %clang --driver-mode=dxc -Tcs_6_1 -Efoo -fcgl  -Fo - %s | FileCheck %s

// Make sure not mangle entry.
// CHECK:define void @foo()
// Make sure add function attribute.
// CHECK:"dx.shader"="compute"
[numthreads(1,1,1)]
void foo() {

}
