// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void throw_builtin() {
    throw 1;
}

// CIR-DAG: cir.global {{.*}} @_ZTIi : !cir.ptr<!u8i>
// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!s32i>, @_ZTIi

// LLVM-DAG: @_ZTIi = external constant ptr
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi, ptr null)

// OGCG-DAG: @_ZTIi = external constant ptr
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi, ptr null)

void throw_bitint() {
    _BitInt(7) a = 0;
    throw a;
}

// CIR-DAG: cir.global {{.*}} @_ZTSDB7_ = #cir.const_array<"DB7_" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5>
// CIR-DAG: cir.global {{.*}} @_ZTIDB7_ = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv123__fundamental_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSDB7_> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!cir.int<s, 7, bitint>>, @_ZTIDB7_

// LLVM-DAG: @_ZTSDB7_ = linkonce_odr global [5 x i8] c"DB7_\00", comdat
// LLVM-DAG: @_ZTIDB7_ = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 16), ptr @_ZTSDB7_ }, comdat
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIDB7_, ptr null)

// OGCG-DAG: @_ZTSDB7_ = linkonce_odr constant [5 x i8] c"DB7_\00", comdat
// OGCG-DAG: @_ZTIDB7_ = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSDB7_ }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIDB7_, ptr null)

void throw_vector() {
  typedef int vi4 __attribute__((vector_size(16)));
  vi4 a;
  throw a;
}

// CIR-DAG: cir.global {{.*}} @_ZTSDv4_i = #cir.const_array<"Dv4_i" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6>
// CIR-DAG: cir.global {{.*}} @_ZTIDv4_i = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv123__fundamental_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSDv4_i> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>, @_ZTIDv4_i

// LLVM-DAG: @_ZTSDv4_i = linkonce_odr global [6 x i8] c"Dv4_i\00", comdat
// LLVM-DAG: @_ZTIDv4_i = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 16), ptr @_ZTSDv4_i }, comdat
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIDv4_i, ptr null)

// OGCG-DAG: @_ZTSDv4_i = linkonce_odr constant [6 x i8] c"Dv4_i\00", comdat
// OGCG-DAG: @_ZTIDv4_i = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSDv4_i }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIDv4_i, ptr null)

void throw_complex() {
    throw __builtin_complex(1.0f, 2.0f);
}

// CIR-DAG: cir.global {{.*}} @_ZTSCf = #cir.const_array<"Cf" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3>
// CIR-DAG: cir.global {{.*}} @_ZTICf = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv123__fundamental_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSCf> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!cir.complex<!cir.float>>, @_ZTICf

// LLVM-DAG: @_ZTSCf = linkonce_odr global [3 x i8] c"Cf\00", comdat
// LLVM-DAG: @_ZTICf = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 16), ptr @_ZTSCf }, comdat
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTICf, ptr null)

// OGCG-DAG: @_ZTSCf = linkonce_odr constant [3 x i8] c"Cf\00", comdat
// OGCG-DAG: @_ZTICf = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSCf }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTICf, ptr null)

void throw_atomic() {
    _Atomic(int) a = 0;
    throw a;
}

// CIR-DAG: cir.global {{.*}} @_ZTSU7_Atomici = #cir.const_array<"U7_Atomici" : !cir.array<!s8i x 10>, trailing_zeros> : !cir.array<!s8i x 11>
// CIR-DAG: cir.global {{.*}} @_ZTIU7_Atomici = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv123__fundamental_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSU7_Atomici> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!s32i>, @_ZTIU7_Atomici

// LLVM-DAG: @_ZTSU7_Atomici = linkonce_odr global [11 x i8] c"U7_Atomici\00", comdat
// LLVM-DAG: @_ZTIU7_Atomici = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 16), ptr @_ZTSU7_Atomici }, comdat
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIU7_Atomici, ptr null)

// OGCG-DAG: @_ZTSU7_Atomici = linkonce_odr constant [11 x i8] c"U7_Atomici\00", comdat
// OGCG-DAG: @_ZTIU7_Atomici = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSU7_Atomici }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIU7_Atomici, ptr null)
