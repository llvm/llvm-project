// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fno-use-cxa-atexit -fapple-kext -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZN5test01aE ={{.*}} global [[A:%.*]] zeroinitializer
// CHECK: @llvm.global_ctors = appending global {{.*}} { i32 65535, void ()* [[CTOR0:@.*]], i8* null }
// CHECK: @llvm.global_dtors = appending global {{.*}} { i32 65535, void ()* [[DTOR0:@.*]], i8* null }

// Check that the base destructor is marked as always_inline when generating
// code for kext.

namespace testBaseDestructor {
#pragma clang optimize off
struct D {
  virtual ~D();
};

D::~D() {}
#pragma clang optimize on
}

// CHECK: define{{.*}} void @_ZN18testBaseDestructor1DD2Ev({{.*}}) unnamed_addr #[[ATTR0:.*]] align 2 {

// CHECK: define{{.*}} void @_ZN18testBaseDestructor1DD1Ev({{.*}}) unnamed_addr #[[ATTR1:.*]] align 2 {

// CHECK: define{{.*}} void @_ZN18testBaseDestructor1DD0Ev({{.*}}) unnamed_addr #[[ATTR1]] align 2 {

// rdar://11241230
namespace test0 {
  struct A { A(); ~A(); };
  A a;
}
// CHECK:    define internal void [[CTOR0_:@.*]]()
// CHECK:      call void @_ZN5test01AC1Ev([[A]]* {{[^,]*}} @_ZN5test01aE)
// CHECK-NEXT: ret void

// CHECK:    define internal void [[CTOR0]]()
// CHECK:      call void [[CTOR0_]]()
// CHECK-NEXT: ret void

// CHECK:    define internal void [[DTOR0]]()
// CHECK:      call void @_ZN5test01AD1Ev([[A]]* @_ZN5test01aE)
// CHECK-NEXT: ret void

// CHECK: attributes #[[ATTR0]] = { alwaysinline nounwind {{.*}} }
// CHECK: attributes #[[ATTR1]] = { noinline nounwind {{.*}} }
