// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fno-use-cxa-atexit -fapple-kext -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fno-use-cxa-atexit -fapple-kext -fcxx-exceptions -fexceptions -DTRY_BODY -emit-llvm -o - %s | FileCheck --check-prefix=TRY %s

// CHECK: @_ZN5test01aE ={{.*}} global [[A:%.*]] zeroinitializer
// CHECK: @llvm.global_ctors = appending global {{.*}} { i32 65535, ptr [[CTOR0:@.*]], ptr null }
// CHECK: @llvm.global_dtors = appending global {{.*}} { i32 65535, ptr [[DTOR0:@.*]], ptr null }

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
namespace test0 {
  struct A { A(); ~A(); };
  A a;
}
// CHECK:    define internal void [[CTOR0_:@.*]]()
// CHECK:      call void @_ZN5test01AC1Ev(ptr {{[^,]*}} @_ZN5test01aE)
// CHECK-NEXT: ret void

// CHECK:    define internal void [[CTOR0]]()
// CHECK:      call void [[CTOR0_]]()
// CHECK-NEXT: ret void

// CHECK:    define internal void [[DTOR0]]()
// CHECK:      call void @_ZN5test01AD1Ev(ptr @_ZN5test01aE)
// CHECK-NEXT: ret void

// CHECK: attributes #[[ATTR0]] = { alwaysinline mustprogress nounwind {{.*}} }
// CHECK: attributes #[[ATTR1]] = { mustprogress noinline nounwind {{.*}} }

#ifdef TRY_BODY
// The complete destructor with a function-try-block contains the body, so it
// is also always_inline.
namespace testTryDestructor {
struct E {
  ~E();
};

E::~E() try {
} catch (...) {
}
}

// TRY: define{{.*}} void @_ZN17testTryDestructor1ED2Ev({{.*}}) unnamed_addr #[[TRY_ATTR:[0-9]+]] align 2 {
// TRY: define{{.*}} void @_ZN17testTryDestructor1ED1Ev({{.*}}) unnamed_addr #[[TRY_ATTR]] align 2 {
// TRY: attributes #[[TRY_ATTR]] = { alwaysinline {{.*}} }
#endif
