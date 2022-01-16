// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fobjc-runtime=ios-6.0 -fobjc-arc \
// RUN:   -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s

id f();
void g() {
  try {
    f();
  } catch (...) {
    f();
  }
}

// CHECK: call noundef i8* @"?f@@YAPAUobjc_object@@XZ"() [ "funclet"(token %1) ]
// CHECK-NEXT: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""() [ "funclet"(token %1) ]

// The corresponding f() call was invoked from the entry basic block.
// CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""(){{$}}
