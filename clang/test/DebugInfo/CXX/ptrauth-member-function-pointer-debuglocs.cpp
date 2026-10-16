// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics \
// RUN:     -emit-llvm -std=c++11 -O1 -disable-llvm-passes \
// RUN:     -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics \
// RUN:     -emit-llvm -std=c++11 -O1 -disable-llvm-passes \
// RUN:     -debug-info-kind=limited %s -o - | FileCheck %s

// Check that compiler-generated *_vfpthunk_ function has a !dbg location
// attached to the call instruction.

// CHECK:      define {{.*}}@_ZN1A2f0Ev_vfpthunk_({{.*}})
// CHECK-SAME:     !dbg ![[SCOPE_INDEX:[0-9]+]]
// CHECK-NOT:  define
// CHECK:        %[[DISCR:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %{{[0-9]+}}, i64 9385)
// CHECK-NOT:  define
// CHECK:        musttail call void %{{[0-9]+}}(ptr
// CHECK-SAME:     [ "ptrauth"(i32 0, i64 %[[DISCR]]) ]
// CHECK-SAME:     !dbg ![[LOCATION_INDEX:[0-9]+]]

// CHECK:      ![[SCOPE_INDEX]] = distinct !DISubprogram(
// CHECK-SAME:                        linkageName: "_ZN1A2f0Ev_vfpthunk_"
// CHECK-SAME:                        flags: DIFlagArtificial | DIFlagThunk
// CHECK:      ![[LOCATION_INDEX]] = !DILocation(line: 0, scope: ![[SCOPE_INDEX]])

volatile long T;

struct A {
  virtual void f0() {
    T = 0;
  }
};
typedef void (A::*MFP)();

void caller() {
  A a;

  MFP x = &A::f0;
  (a.*x)();
}
