// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll  %s

class A {
public:
  int a;
  virtual void v() {}
};

class B : public virtual A {
public:
  int b;
  virtual void w();
};

class C : public virtual A {
public:
  long c;
  virtual void x() {}
};

class D : public B, public C {
public:
  long d;
  virtual void y() {}
};


int f() {
  B *b = new D ();
  return 0;
}

// Vtable of Class A
// CIR: cir.global constant linkonce_odr @_ZTV1A = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !rec_anon_struct3 {alignment = 8 : i64}

// Class A constructor
// CIR: cir.func {{.*}} @_ZN1AC2Ev(%arg0: !cir.ptr<!rec_A>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{.*}} : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR: }

// Vtable of Class D
// CIR: cir.global constant linkonce_odr @_ZTV1D = #cir.vtable<{#cir.const_array<[#cir.ptr<40 : i64> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1D1yEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>, #cir.const_array<[#cir.ptr<24 : i64> : !cir.ptr<!u8i>, #cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>, #cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<-40 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !rec_anon_struct4 {alignment = 8 : i64}
// VTT of class D
// CIR: cir.global constant linkonce_odr @_ZTT1D = #cir.const_array<[#cir.global_view<@_ZTV1D, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D0_1B, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D0_1B, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D16_1C, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D16_1C, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTV1D, [2 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTV1D, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 7> {alignment = 8 : i64}

// Class B constructor
// CIR: cir.func {{.*}} @_ZN1BC2Ev(%arg0: !cir.ptr<!rec_B>
// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>

// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{.*}} : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.vptr -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.const #cir.int<-24> : !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride %{{[0-9]+}}, %{{[0-9]+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   %{{[0-9]+}} = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!s64i>, !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride %{{[0-9]+}}, %{{[0-9]+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR: }

// LLVM-LABEL: @_ZN1BC2Ev
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]], align 8
// LLVM:   %[[V:.*]] = load ptr, ptr %[[VTT]], align 8
// LLVM:   store ptr %[[V]], ptr %[[THIS]], align 8
// LLVM:   getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM:   ret void
// LLVM: }

// Class C constructor
// CIR: cir.func {{.*}} @_ZN1CC2Ev(%arg0: !cir.ptr<!rec_C>
// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>

// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.vptr -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.const #cir.int<-24> : !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride %{{[0-9]+}}, %{{[0-9]+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:   %{{[0-9]+}} = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!s64i>, !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride %{{[0-9]+}}, %{{[0-9]+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast bitcast %{{[0-9]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_C>
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR: }

// Class D constructor
// CIR: cir.func {{.*}} @_ZN1DC1Ev(%arg0: !cir.ptr<!rec_D>
// CIR:   %{{[0-9]+}} = cir.alloca !cir.ptr<!rec_D>, !cir.ptr<!cir.ptr<!rec_D>>, ["this", init] {alignment = 8 : i64}
// CIR:   cir.store{{.*}} %arg0, %{{[0-9]+}} : !cir.ptr<!rec_D>, !cir.ptr<!cir.ptr<!rec_D>>
// CIR:   %[[D_PTR:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!rec_D>>, !cir.ptr<!rec_D>
// CIR:   %[[A_PTR:.*]] = cir.base_class_addr %[[D_PTR]] : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR:   cir.call @_ZN1AC2Ev(%[[A_PTR]]) : (!cir.ptr<!rec_A>) -> ()

// CIR:   %[[B_PTR:.*]] = cir.base_class_addr %[[D_PTR]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_B>
// CIR:   %[[VTT_D_TO_B:.*]] = cir.vtt.address_point @_ZTT1D, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.call @_ZN1BC2Ev(%[[B_PTR]], %[[VTT_D_TO_B]]) : (!cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!void>>) -> ()

// CIR:   %[[C_PTR:.*]] = cir.base_class_addr %1 : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:   %[[VTT_D_TO_C:.*]] = cir.vtt.address_point @_ZTT1D, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.call @_ZN1CC2Ev(%[[C_PTR]], %[[VTT_D_TO_C]]) : (!cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!void>>) -> ()

// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, address_point = <index = 2, offset = 3>) : !cir.vptr

// CIR:   %{{[0-9]+}} = cir.base_class_addr %{{[0-9]+}} : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, address_point = <index = 1, offset = 3>) : !cir.vptr

// CIR:   cir.base_class_addr %{{[0-9]+}} : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:   %{{[0-9]+}} = cir.vtable.get_vptr %{{[0-9]+}} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %{{[0-9]+}}, %{{[0-9]+}} : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   cir.return
// CIR: }

// Note: GEP emitted by cir might not be the same as LLVM, due to constant folding.
// LLVM-LABEL: @_ZN1DC1Ev
// LLVM:   %2 = alloca ptr, i64 1, align 8
// LLVM:   store ptr %0, ptr %2, align 8
// LLVM:   %[[THIS:.*]] = load ptr, ptr %2, align 8
// LLVM:   %[[BASE_A:.*]] = getelementptr i8, ptr %[[THIS]], i32 40
// LLVM:   call void @_ZN1AC2Ev(ptr %[[BASE_A]])
// LLVM:   call void @_ZN1BC2Ev(ptr %[[THIS]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 8))
// LLVM:   %[[BASE_C:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM:   call void @_ZN1CC2Ev(ptr %[[BASE_C]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 24))
// LLVM:   ret void
// LLVM: }

namespace other {
  struct A {
    A();
    ~A();
  };

  struct B : virtual A {
    B();
    ~B();
  };

  extern int foo();
  B::B() {
    int x = foo();
  }

  B::~B() {
    int y = foo();
  }
}

// CIR-LABEL:   cir.func {{.*}} @_ZN5other1BD1Ev(
// CIR-SAME:                               %[[VAL_0:.*]]: !cir.ptr<!rec_other3A3AB>
// CIR:           %[[VAL_1:.*]] = cir.alloca !cir.ptr<!rec_other3A3AB>, !cir.ptr<!cir.ptr<!rec_other3A3AB>>, ["this", init] {alignment = 8 : i64}
// CIR:           cir.store{{.*}} %[[VAL_0]], %[[VAL_1]] : !cir.ptr<!rec_other3A3AB>, !cir.ptr<!cir.ptr<!rec_other3A3AB>>
// CIR:           %[[VAL_2:.*]] = cir.load{{.*}} %[[VAL_1]] : !cir.ptr<!cir.ptr<!rec_other3A3AB>>, !cir.ptr<!rec_other3A3AB>
// CIR:           %[[VAL_3:.*]] = cir.vtt.address_point @_ZTTN5other1BE, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:           cir.call @_ZN5other1BD2Ev(%[[VAL_2]], %[[VAL_3]]) : (!cir.ptr<!rec_other3A3AB>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR:           %[[VAL_4:.*]] = cir.base_class_addr %[[VAL_2]] : !cir.ptr<!rec_other3A3AB> nonnull [0] -> !cir.ptr<!rec_other3A3AA>
// CIR:           cir.call @_ZN5other1AD2Ev(%[[VAL_4]]) : (!cir.ptr<!rec_other3A3AA>) -> ()
// CIR:           cir.return
// CIR:         }
