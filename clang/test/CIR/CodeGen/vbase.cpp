// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Test the record layout for a class with a primary virtual base.
class Base {
public:
  virtual void f();
};

class Derived : public virtual Base {};

// This is just here to force the record types to be emitted.
void f() {
  Derived d;
}

// CIR: !rec_Base = !cir.record<class "Base" {!cir.vptr}>
// CIR: !rec_Derived = !cir.record<class "Derived" {!rec_Base}>

// LLVM: %class.Derived = type { %class.Base }
// LLVM: %class.Base = type { ptr }

// OGCG: %class.Derived = type { %class.Base }
// OGCG: %class.Base = type { ptr }

// Test the constructor handling for a class with a virtual base.
struct A {
  int a;
};

struct B:  virtual A {
  int b;
};

void ppp() { B b; }

// Note: OGCG speculatively emits the VTT and VTables. This is not yet implemented in CIR.

// Vtable definition for B
// CIR:  cir.global "private" external @_ZTV1B

// LLVM: @_ZTV1B = external global { [3 x ptr] }

// OGCG: @_ZTV1B = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr inttoptr (i64 12 to ptr), ptr null, ptr @_ZTI1B] }, comdat, align 8

// Constructor for A
// CIR: cir.func comdat linkonce_odr @_ZN1AC2Ev(%arg0: !cir.ptr<!rec_A>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:   cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:   cir.return

// LLVM: define{{.*}} void @_ZN1AC2Ev(ptr %[[THIS_ARG:.*]]) {
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   ret void

// Note: OGCG elides the constructor for A. This is not yet implemented in CIR.

// Constructor for B
// CIR: cir.func comdat linkonce_odr @_ZN1BC1Ev(%arg0: !cir.ptr<!rec_B>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["this", init]
// CIR:   cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:   %[[BASE_A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_B> nonnull [12] -> !cir.ptr<!rec_A>
// CIR:   cir.call @_ZN1AC2Ev(%[[BASE_A_ADDR]]) nothrow : (!cir.ptr<!rec_A>) -> ()
// CIR:   %[[VTABLE:.*]] = cir.vtable.address_point(@_ZTV1B, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR:   %[[B_VPTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store align(8) %[[VTABLE]], %[[B_VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   cir.return

// LLVM: define{{.*}} void @_ZN1BC1Ev(ptr %[[THIS_ARG:.*]]) {
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[BASE_A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 12
// LLVM:   call void @_ZN1AC2Ev(ptr %[[BASE_A_ADDR]])
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 24), ptr %[[THIS]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_ZN1BC1Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[BASE_A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 12
// OGCG:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 3), ptr %[[THIS]]
// OGCG:   ret void

