// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-cir -fcxx-exceptions -fexceptions -mmlir --mlir-print-ir-before=cir-cxxabi-lowering -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM,OGCG
#include <typeinfo>

struct Struct {
};

// PTR:
// CIR-DAG: cir.global constant external @_ZTIP6Struct = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv119__pointer_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSP6Struct> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTI6Struct> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global "private" external @_ZTVN10__cxxabiv119__pointer_type_infoE : !cir.ptr<!cir.ptr<!u8i>> {alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr @_ZTSP6Struct = #cir.const_array<"P6Struct" : !cir.array<!s8i x 8>, trailing_zeros> : !cir.array<!s8i x 9> {alignment = 1 : i64}
// CIR-DAG: cir.global constant external @_ZTI6Struct = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS6Struct> : !cir.ptr<!u8i>}> : !rec_anon_struct {alignment = 8 : i64}
// CIR-DAG: cir.global "private" external @_ZTVN10__cxxabiv117__class_type_infoE : !cir.ptr<!cir.ptr<!u8i>> {alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr @_ZTS6Struct = #cir.const_array<"6Struct" : !cir.array<!s8i x 7>, trailing_zeros> : !cir.array<!s8i x 8> {alignment = 1 : i64}
//
// LLVM-CIR-DAG: @_ZTIP6Struct = constant { ptr, ptr, i32, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP6Struct, i32 0, ptr @_ZTI6Struct }, align 8
// OGCG-DAG: @_ZTIP6Struct = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSP6Struct, i32 0, ptr @_ZTI6Struct }, align 8
// LLVM-DAG: @_ZTVN10__cxxabiv119__pointer_type_infoE = external global 
// LLVM-DAG: @_ZTSP6Struct = linkonce_odr{{.*}} [9 x i8] c"P6Struct\00", align 1
// LLVM-CIR-DAG: @_ZTI6Struct = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS6Struct }, align 8
// OGCG-DAG: @_ZTI6Struct = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS6Struct }, align 8
// LLVM-DAG: @_ZTVN10__cxxabiv117__class_type_infoE = external global
// LLVM-DAG: @_ZTS6Struct = linkonce_odr{{.*}} [8 x i8] c"6Struct\00", align 1

// MemFnPtr:
// CIR-DAG: cir.global constant external @_ZTIM6StructFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSM6StructFvvE> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTIFvvE> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI6Struct> : !cir.ptr<!u8i>}> : !{{.*}} {alignment = 8 : i64}
// CIR-DAG: cir.global "private" external @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE : !cir.ptr<!cir.ptr<!u8i>> {alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr @_ZTSM6StructFvvE = #cir.const_array<"M6StructFvvE" : !cir.array<!s8i x 12>, trailing_zeros> : !cir.array<!s8i x 13> {alignment = 1 : i64}
// CIR-DAG: cir.global constant external @_ZTIFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv120__function_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSFvvE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global "private" external @_ZTVN10__cxxabiv120__function_type_infoE : !cir.ptr<!cir.ptr<!u8i>> {alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr @_ZTSFvvE = #cir.const_array<"FvvE" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5> {alignment = 1 : i64}
//
// LLVM-CIR-DAG: @_ZTIM6StructFvvE = constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 16), ptr @_ZTSM6StructFvvE, i32 0, ptr @_ZTIFvvE, ptr @_ZTI6Struct }, align 8
// OGCG-DAG: @_ZTIM6StructFvvE = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2), ptr @_ZTSM6StructFvvE, i32 0, ptr @_ZTIFvvE, ptr @_ZTI6Struct }, align 8
// LLVM-DAG: @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE = external global
// LLVM-DAG: @_ZTSM6StructFvvE = linkonce_odr{{.*}}[13 x i8] c"M6StructFvvE\00", align 1
// LLVM-CIR-DAG: @_ZTIFvvE = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 16), ptr @_ZTSFvvE }, align 8
// OGCG-DAG: @_ZTIFvvE = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 2), ptr @_ZTSFvvE }, align 8
// LLVM-DAG: @_ZTVN10__cxxabiv120__function_type_infoE = external global
// LLVM-DAG: @_ZTSFvvE = linkonce_odr{{.*}}[5 x i8] c"FvvE\00", align 1

// MemDataPtr:
// CIR-DAG: cir.global constant external @_ZTIM6Structi = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSM6Structi> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI6Struct> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr @_ZTSM6Structi = #cir.const_array<"M6Structi" : !cir.array<!s8i x 9>, trailing_zeros> : !cir.array<!s8i x 10> {alignment = 1 : i64}
// CIR-DAG: cir.global "private" constant external @_ZTIi : !cir.ptr<!u8i>
//
// LLVM-CIR-DAG:@_ZTIM6Structi = constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 16), ptr @_ZTSM6Structi, i32 0, ptr @_ZTIi, ptr @_ZTI6Struct }, align 8
//
// OGCG-DAG:@_ZTIM6Structi = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2), ptr @_ZTSM6Structi, i32 0, ptr @_ZTIi, ptr @_ZTI6Struct }, align 8
// LLVM-DAG: @_ZTSM6Structi = linkonce_odr{{.*}}[10 x i8] c"M6Structi\00", align 1
// LLVM-DAG: @_ZTIi = external constant ptr

auto ptr() {
  return typeid(Struct*);
  // CIR: cir.get_global @_ZTIP6Struct : !cir.ptr<!{{.*}}>
  // LLVM: call void @_ZNSt9type_infoC1ERKS_({{.*}}@_ZTIP6Struct)
}


using MemFnPtr = void (Struct::*)(void);
auto memFnPtr() {
  return typeid(MemFnPtr);
  // CIR: cir.get_global @_ZTIM6StructFvvE : !cir.ptr<!{{.*}}>
  // LLVM: call void @_ZNSt9type_infoC1ERKS_({{.*}}@_ZTIM6StructFvvE)
}

using MemDataPtr = int Struct::*;
auto memDataPtr() {
  return typeid(MemDataPtr);
  // CIR: cir.get_global @_ZTIM6Structi : !cir.ptr<!{{.*}}>
  // LLVM: call void @_ZNSt9type_infoC1ERKS_({{.*}}@_ZTIM6Structi)
}
