// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// Test that throwing a pointer to a noexcept function produces correct RTTI
// with the PTI_Noexcept flag (0x40 = 64) set in the __pointer_type_info.

void f() noexcept {
  throw f;
}

// The pointee type _ZTIFvvE (function type info for void()) must be emitted
// using the __function_type_info vtable.
// CIR-DAG: cir.global {{.*}} @_ZTSFvvE = #cir.const_array<"FvvE" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5>
// CIR-DAG: cir.global {{.*}} @_ZTIFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv120__function_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSFvvE> : !cir.ptr<!u8i>}>

// The pointer type info _ZTIPDoFvvE must include flag 64 (PTI_Noexcept).
// CIR-DAG: cir.global {{.*}} @_ZTSPDoFvvE = #cir.const_array<"PDoFvvE" : !cir.array<!s8i x 7>, trailing_zeros> : !cir.array<!s8i x 8>
// CIR-DAG: cir.global {{.*}} @_ZTIPDoFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv119__pointer_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSPDoFvvE> : !cir.ptr<!u8i>, #cir.int<64> : !u32i, #cir.global_view<@_ZTIFvvE> : !cir.ptr<!u8i>}>

// CIR: cir.throw %{{.*}} : !cir.ptr<!cir.ptr<!cir.func<()>>>, @_ZTIPDoFvvE

// LLVM-DAG: @_ZTSFvvE = linkonce_odr constant [5 x i8] c"FvvE\00", comdat
// LLVM-DAG: @_ZTSPDoFvvE = linkonce_odr constant [8 x i8] c"PDoFvvE\00", comdat

// CIR and OGCG represent the reference to the type info vtable differently,
// as a byte offset from the vtable symbol or as an index into an array of
// pointers, but both name its address point.
// LLVMCIR-DAG: @_ZTIFvvE = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 16), ptr @_ZTSFvvE }, comdat
// LLVMCIR-DAG: @_ZTIPDoFvvE = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSPDoFvvE, i32 64, ptr @_ZTIFvvE }, comdat
// OGCG-DAG: @_ZTIFvvE = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 2), ptr @_ZTSFvvE }, comdat
// OGCG-DAG: @_ZTIPDoFvvE = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSPDoFvvE, i32 64, ptr @_ZTIFvvE }, comdat

// The throw unwinds to the terminate scope of the enclosing noexcept function.
// LLVM-LABEL: define{{.*}} void @_Z1fv()
// LLVM-SAME:    personality ptr @__gxx_personality_v0
// LLVM:         invoke void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIPDoFvvE, ptr null)
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:      catch ptr null
// LLVM:         call void @__clang_call_terminate(ptr %{{.*}})
// LLVM-NEXT:    unreachable
