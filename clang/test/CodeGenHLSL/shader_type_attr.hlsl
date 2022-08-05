// RUN: %clang --driver-mode=dxc -Tlib_6_x -fcgl -Fo - %s | FileCheck %s

// Make sure not mangle entry.
// CHECK:define void @foo()
// Make sure add function attribute.
// CHECK:"dx.shader"="compute"
[shader("compute")]
[numthreads(1,1,1)]
void foo() {

}
