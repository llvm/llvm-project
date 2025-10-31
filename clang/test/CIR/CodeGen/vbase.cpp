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

void f() {
  Derived d;
  d.f();
}

class DerivedFinal final : public virtual Base {};

void g() {
  DerivedFinal df;
  df.f();
}

// CIR: !rec_Base = !cir.record<class "Base" {!cir.vptr}>
// CIR: !rec_Derived = !cir.record<class "Derived" {!rec_Base}>
// CIR: !rec_DerivedFinal = !cir.record<class "DerivedFinal" {!rec_Base}>

// LLVM: %class.Derived = type { %class.Base }
// LLVM: %class.Base = type { ptr }
// LLVM: %class.DerivedFinal = type { %class.Base }

// OGCG: %class.Derived = type { %class.Base }
// OGCG: %class.Base = type { ptr }
// OGCG: %class.DerivedFinal = type { %class.Base }

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

// CIR: cir.func {{.*}}@_Z1fv()
// CIR:   %[[D:.+]] = cir.alloca !rec_Derived, !cir.ptr<!rec_Derived>, ["d", init]
// CIR:   cir.call @_ZN7DerivedC1Ev(%[[D]]) nothrow : (!cir.ptr<!rec_Derived>) -> ()
// CIR:   %[[VPTR_PTR:.+]] = cir.vtable.get_vptr %[[D]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR:.+]] = cir.load {{.*}} %[[VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[VPTR_I8:.+]] = cir.cast bitcast %[[VPTR]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR:   %[[NEG32:.+]] = cir.const #cir.int<-32> : !s64i
// CIR:   %[[ADJ_VPTR_I8:.+]] = cir.ptr_stride %[[VPTR_I8]], %[[NEG32]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[OFFSET_PTR:.+]] = cir.cast bitcast %[[ADJ_VPTR_I8]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   %[[OFFSET:.+]] = cir.load {{.*}} %[[OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR:   %[[D_I8:.+]] = cir.cast bitcast %[[D]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!u8i>
// CIR:   %[[ADJ_THIS_I8:.+]] = cir.ptr_stride %[[D_I8]], %[[OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[ADJ_THIS_D:.+]] = cir.cast bitcast %[[ADJ_THIS_I8]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_Derived>
// CIR:   %[[BASE_THIS:.+]] = cir.cast bitcast %[[ADJ_THIS_D]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!rec_Base>
// CIR:   %[[BASE_VPTR_PTR:.+]] = cir.vtable.get_vptr %[[BASE_THIS]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// CIR:   %[[BASE_VPTR:.+]] = cir.load {{.*}} %[[BASE_VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[SLOT_PTR:.+]] = cir.vtable.get_virtual_fn_addr %[[BASE_VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>
// CIR:   %[[FN:.+]] = cir.load {{.*}} %[[SLOT_PTR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>
// CIR:   cir.call %[[FN]](%[[BASE_THIS]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>, !cir.ptr<!rec_Base>) -> ()
// CIR:   cir.return

// CIR: cir.func {{.*}}@_Z1gv()
// CIR:   %[[DF:.+]] = cir.alloca !rec_DerivedFinal, !cir.ptr<!rec_DerivedFinal>, ["df", init]
// CIR:   cir.call @_ZN12DerivedFinalC1Ev(%[[DF]]) nothrow : (!cir.ptr<!rec_DerivedFinal>) -> ()
// CIR:   %[[BASE_THIS_2:.+]] = cir.base_class_addr %[[DF]] : !cir.ptr<!rec_DerivedFinal> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[BASE_VPTR_PTR_2:.+]] = cir.vtable.get_vptr %[[BASE_THIS_2]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// CIR:   %[[BASE_VPTR_2:.+]] = cir.load {{.*}} %[[BASE_VPTR_PTR_2]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[SLOT_PTR_2:.+]] = cir.vtable.get_virtual_fn_addr %[[BASE_VPTR_2]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>
// CIR:   %[[FN_2:.+]] = cir.load {{.*}} %[[SLOT_PTR_2]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>
// CIR:   cir.call %[[FN_2]](%[[BASE_THIS_2]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>, !cir.ptr<!rec_Base>) -> ()
// CIR:   cir.return

// LLVM: define {{.*}}void @_Z1fv(){{.*}}
// LLVM:   %[[D:.+]] = alloca {{.*}}
// LLVM:   call void @_ZN7DerivedC1Ev(ptr %[[D]])
// LLVM:   %[[VPTR_ADDR:.+]] = load ptr, ptr %[[D]]
// LLVM:   %[[NEG32_PTR:.+]] = getelementptr i8, ptr %[[VPTR_ADDR]], i64 -32
// LLVM:   %[[OFF:.+]] = load i64, ptr %[[NEG32_PTR]]
// LLVM:   %[[ADJ_THIS:.+]] = getelementptr i8, ptr %[[D]], i64 %[[OFF]]
// LLVM:   %[[VFN_TAB:.+]] = load ptr, ptr %[[ADJ_THIS]]
// LLVM:   %[[SLOT0:.+]] = getelementptr inbounds ptr, ptr %[[VFN_TAB]], i32 0
// LLVM:   %[[VFN:.+]] = load ptr, ptr %[[SLOT0]]
// LLVM:   call void %[[VFN]](ptr %[[ADJ_THIS]])
// LLVM:   ret void

// LLVM: define {{.*}}void @_Z1gv(){{.*}}
// LLVM:   %[[DF:.+]] = alloca {{.*}}
// LLVM:   call void @_ZN12DerivedFinalC1Ev(ptr %[[DF]])
// LLVM:   %[[VPTR2:.+]] = load ptr, ptr %[[DF]]
// LLVM:   %[[SLOT0_2:.+]] = getelementptr inbounds ptr, ptr %[[VPTR2]], i32 0
// LLVM:   %[[VFN2:.+]] = load ptr, ptr %[[SLOT0_2]]
// LLVM:   call void %[[VFN2]](ptr %[[DF]])
// LLVM:   ret void

// OGCG: define {{.*}}void @_Z1fv()
// OGCG:   %[[D:.+]] = alloca {{.*}}
// OGCG:   call void @_ZN7DerivedC1Ev(ptr {{.*}} %[[D]])
// OGCG:   %[[VTABLE:.+]] = load ptr, ptr %[[D]]
// OGCG:   %[[NEG32_PTR:.+]] = getelementptr i8, ptr %[[VTABLE]], i64 -32
// OGCG:   %[[OFF:.+]] = load i64, ptr %[[NEG32_PTR]]
// OGCG:   %[[ADJ_THIS:.+]] = getelementptr inbounds i8, ptr %[[D]], i64 %[[OFF]]
// OGCG:   call void @_ZN4Base1fEv(ptr {{.*}} %[[ADJ_THIS]])
// OGCG:   ret void

// OGCG: define {{.*}}void @_Z1gv()
// OGCG:   %[[DF:.+]] = alloca {{.*}}
// OGCG:   call void @_ZN12DerivedFinalC1Ev(ptr {{.*}} %[[DF]])
// OGCG:   call void @_ZN4Base1fEv(ptr {{.*}} %[[DF]])
// OGCG:   ret void

// Constructor for B
// CIR: cir.func comdat linkonce_odr @_ZN1BC1Ev(%arg0: !cir.ptr<!rec_B>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["this", init]
// CIR:   cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:   %[[BASE_A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_B> nonnull [12] -> !cir.ptr<!rec_A>
// CIR:   %[[VTABLE:.*]] = cir.vtable.address_point(@_ZTV1B, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR:   %[[B_VPTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store align(8) %[[VTABLE]], %[[B_VPTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   cir.return

// LLVM: define{{.*}} void @_ZN1BC1Ev(ptr %[[THIS_ARG:.*]]){{.*}} {
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[BASE_A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 12
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 24), ptr %[[THIS]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_ZN1BC1Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[BASE_A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 12
// OGCG:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 3), ptr %[[THIS]]
// OGCG:   ret void
