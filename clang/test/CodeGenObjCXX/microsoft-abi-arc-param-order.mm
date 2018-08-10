// RUN: %clang_cc1 -mconstructor-aliases -fobjc-arc -triple i686-pc-win32 -emit-llvm -o - %s | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
  int a;
};

// Verify that we destruct things from left to right in the MS C++ ABI: a, b, c, d.
//
// CHECK-LABEL: define dso_local void @"?test_arc_order@@YAXUA@@PAU.objc_object@@01@Z"
// CHECK:                       (<{ %struct.A, i8*, %struct.A, i8* }>* inalloca)
void test_arc_order(A a, id __attribute__((ns_consumed)) b , A c, id __attribute__((ns_consumed)) d) {
  // CHECK: call x86_thiscallcc void @"??1A@@QAE@XZ"(%struct.A* %{{.*}})
  // CHECK: call void @objc_storeStrong(i8** %{{.*}}, i8* null)
  // CHECK: call x86_thiscallcc void @"??1A@@QAE@XZ"(%struct.A* %{{.*}})
  // CHECK: call void @objc_storeStrong(i8** %{{.*}}, i8* null)
  // CHECK: ret void
}
