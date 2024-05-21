// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

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


int main() {
    B *b = new D ();
    return 0;
}

// Vtable of Class A
// CIR: cir.global linkonce_odr @_ZTV1A = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !ty_anon_struct3 {alignment = 8 : i64}

// Class A constructor
// CIR: cir.func linkonce_odr @_ZN1AC2Ev(%arg0: !cir.ptr<!ty_A>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1A, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_A>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR: }

// Vtable of Class D
// CIR: cir.global linkonce_odr @_ZTV1D = #cir.vtable<{#cir.const_array<[#cir.ptr<40 : i64> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1D1yEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>, #cir.const_array<[#cir.ptr<24 : i64> : !cir.ptr<!u8i>, #cir.ptr<-16 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>, #cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<-40 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !ty_anon_struct4 {alignment = 8 : i64}
// VTT of class D
// CIR: cir.global linkonce_odr @_ZTT1D = #cir.const_array<[#cir.global_view<@_ZTV1D, [0 : i32, 0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D0_1B, [0 : i32, 0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D0_1B, [0 : i32, 1 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D16_1C, [0 : i32, 0 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTC1D16_1C, [0 : i32, 1 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTV1D, [0 : i32, 2 : i32, 3 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTV1D, [0 : i32, 1 : i32, 3 : i32]> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 7> {alignment = 8 : i64}

// Class B constructor
// CIR: cir.func linkonce_odr @_ZN1BC2Ev(%arg0: !cir.ptr<!ty_B> loc({{.*}}), %arg1: !cir.ptr<!cir.ptr<!void>> loc({{.*}})) extra(#fn_attr) {
// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_B>), !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>

// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_B>), !cir.ptr<!cir.ptr<!u8i>>
// CIR:   %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.const #cir.int<-24> : !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:   %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!s64i>, !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_B>), !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: }

// Class C constructor
// CIR: cir.func linkonce_odr @_ZN1CC2Ev(%arg0: !cir.ptr<!ty_C> loc({{.*}}), %arg1: !cir.ptr<!cir.ptr<!void>> loc({{.*}})) extra(#fn_attr) {
// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_C>), !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>

// CIR:   %{{[0-9]+}} = cir.vtt.address_point %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   %{{[0-9]+}} = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_C>), !cir.ptr<!cir.ptr<!u8i>>
// CIR:   %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.const #cir.int<-24> : !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:   %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!s64i>, !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_C>), !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: }

// Class D constructor
// CIR: cir.func linkonce_odr @_ZN1DC1Ev(%arg0: !cir.ptr<!ty_D> loc({{.*}})) extra(#fn_attr) {
// CIR:   %{{[0-9]+}} = cir.alloca !cir.ptr<!ty_D>, !cir.ptr<!cir.ptr<!ty_D>>, ["this", init] {alignment = 8 : i64}
// CIR:   cir.store %arg0, %{{[0-9]+}} : !cir.ptr<!ty_D>, !cir.ptr<!cir.ptr<!ty_D>>
// CIR:   %[[D_PTR:.*]] = cir.load %0 : !cir.ptr<!cir.ptr<!ty_D>>, !cir.ptr<!ty_D>
// CIR:   %[[A_PTR:.*]] = cir.base_class_addr(%[[D_PTR]] : !cir.ptr<!ty_D> nonnull) [40] -> !cir.ptr<!ty_A>
// CIR:   cir.call @_ZN1AC2Ev(%[[A_PTR]]) : (!cir.ptr<!ty_A>) -> ()

// CIR:   %[[B_PTR:.*]] = cir.base_class_addr(%[[D_PTR]] : !cir.ptr<!ty_D> nonnull) [0] -> !cir.ptr<!ty_B>
// CIR:   %[[VTT_D_TO_B:.*]] = cir.vtt.address_point @_ZTT1D, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.call @_ZN1BC2Ev(%[[B_PTR]], %[[VTT_D_TO_B]]) : (!cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!void>>) -> ()

// CIR:   %[[C_PTR:.*]] = cir.base_class_addr(%1 : !cir.ptr<!ty_D> nonnull) [16] -> !cir.ptr<!ty_C>
// CIR:   %[[VTT_D_TO_C:.*]] = cir.vtt.address_point @_ZTT1D, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.call @_ZN1CC2Ev(%[[C_PTR]], %[[VTT_D_TO_C]]) : (!cir.ptr<!ty_C>, !cir.ptr<!cir.ptr<!void>>) -> ()

// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, vtable_index = 0, address_point_index = 3) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_D>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, vtable_index = 2, address_point_index = 3) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>

// CIR:   %{{[0-9]+}} = cir.const #cir.int<40> : !s64i
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_D>), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!u8i>), !cir.ptr<!ty_D>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_D>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV1D, vtable_index = 1, address_point_index = 3) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>

// CIR:   %{{[0-9]+}} = cir.const #cir.int<16> : !s64i
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_D>), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!u8i>, %{{[0-9]+}} : !s64i), !cir.ptr<!u8i>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!u8i>), !cir.ptr<!ty_D>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_D>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.return
// CIR: }