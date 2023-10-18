// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct A {
  int a;
};

struct B:  virtual A {
  int b;
};

void ppp() { B b; }


// Vtable definition for B
// CHECK:  cir.global linkonce_odr @_ZTV1B = #cir.vtable<{#cir.const_array<[#cir.ptr<12> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}>

// VTT for B.
// CHECK:  cir.global linkonce_odr @_ZTT1B = #cir.const_array<[#cir.global_view<@_ZTV1B, [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<3> : !s32i]> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 1>

// CHECK:  cir.global "private" external @_ZTVN10__cxxabiv121__vmi_class_type_infoE

// Type info name for B
// CHECK:  cir.global linkonce_odr @_ZTS1B = #cir.const_array<"1B" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>

// CHECK:  cir.global "private" external @_ZTVN10__cxxabiv117__class_type_infoE : !cir.ptr<!cir.ptr<!u8i>>

// Type info name for A
// CHECK:  cir.global linkonce_odr @_ZTS1A = #cir.const_array<"1A" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>

// Type info A.
// CHECK:  cir.global constant external @_ZTI1A = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [#cir.int<2> : !s64i]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1A> : !cir.ptr<!u8i>}>

// Type info B.
// CHECK:  cir.global constant external @_ZTI1B = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv121__vmi_class_type_infoE, [#cir.int<2> : !s64i]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1B> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.int<1> : !u32i, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>, #cir.int<-6141> : !s64i}>
