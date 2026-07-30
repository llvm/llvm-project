// RUN: %clang_cc1 -mconstructor-aliases -fobjc-arc -triple i686-pc-win32 -emit-llvm -o - %s | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
  int a;
};

// Verify that we destruct things from left to right in the MS C++ ABI: a, b, c, d.
//
// CHECK-LABEL: define dso_local void @"?test_arc_order@@YAXUA@@PAUobjc_object@@01@Z"
// CHECK:                       (ptr inalloca(<{ %struct.A, ptr, %struct.A, ptr }>) %0)
void test_arc_order(A a, id __attribute__((ns_consumed)) b , A c, id __attribute__((ns_consumed)) d) {
  // CHECK: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK: call void @llvm.objc.storeStrong(ptr %{{.*}}, ptr null)
  // CHECK: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK: call void @llvm.objc.storeStrong(ptr %{{.*}}, ptr null)
  // CHECK: ret void
}
