// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test RTTI emission for pointer-to-member types (abi::__pointer_to_member_type_info).
// Each descriptor must use the __pointer_to_member_type_info vtable and carry
// three extra fields beyond the standard type_info header:
//   __flags    (unsigned int)
//   __pointee  (const type_info* for the pointed-to type)
//   __context  (const __class_type_info* for the containing class)

struct A {
  int data;
  void func();
};

// --- Test 1: data member pointer (int A::*) ---

void throw_data_member_ptr() {
  int A::*p = &A::data;
  throw p;
}

// Class A has no bases, so it uses __class_type_info.
// CIR-DAG: cir.global {{.*}} @_ZTS1A = #cir.const_array<"1A" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>
// CIR-DAG: cir.global {{.*}} @_ZTI1A = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1A> : !cir.ptr<!u8i>}>

// The type name for "int A::*" is "M1Ai".
// CIR-DAG: cir.global {{.*}} @_ZTSM1Ai = #cir.const_array<"M1Ai" : !cir.array<!s8i x 4>> : !cir.array<!s8i x 4>

// The member-pointer type info must use the __pointer_to_member_type_info vtable,
// flags=0 (int has no cv-qualifiers), pointee=int (_ZTIi), context=A (_ZTI1A).
// CIR-DAG: cir.global {{.*}} @_ZTIM1Ai = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSM1Ai> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>}>

// CIR-DAG: cir.throw %{{.*}} : !cir.ptr<!s64i>, @_ZTIM1Ai

// LLVM-DAG: @_ZTSM1Ai = linkonce_odr global [4 x i8] c"M1Ai"
// LLVM-DAG: @_ZTS1A = linkonce_odr global [2 x i8] c"1A"
// LLVM-DAG: @_ZTI1A = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1A }
// LLVM-DAG: @_ZTIM1Ai = constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 16), ptr @_ZTSM1Ai, i32 0, ptr @_ZTIi, ptr @_ZTI1A }
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIM1Ai, ptr null)

// OGCG-DAG: @_ZTSM1Ai = linkonce_odr constant [5 x i8] c"M1Ai\00", comdat
// OGCG-DAG: @_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", comdat
// OGCG-DAG: @_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat
// OGCG-DAG: @_ZTIM1Ai = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2), ptr @_ZTSM1Ai, i32 0, ptr @_ZTIi, ptr @_ZTI1A }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIM1Ai, ptr null)

// --- Test 2: member function pointer (void (A::*)()) ---

void throw_member_fn_ptr() {
  void (A::*p)() = &A::func;
  throw p;
}

// The type name for "void (A::*)()" is "M1AFvvE".
// CIR-DAG: cir.global {{.*}} @_ZTSM1AFvvE = #cir.const_array<"M1AFvvE" : !cir.array<!s8i x 7>> : !cir.array<!s8i x 7>

// The pointee "void()" is a function type, using __function_type_info.
// CIR-DAG: cir.global {{.*}} @_ZTSFvvE = #cir.const_array<"FvvE" : !cir.array<!s8i x 4>> : !cir.array<!s8i x 4>
// CIR-DAG: cir.global {{.*}} @_ZTIFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv120__function_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSFvvE> : !cir.ptr<!u8i>}>

// flags=0, pointee=void() (_ZTIFvvE), context=A (_ZTI1A).
// CIR-DAG: cir.global {{.*}} @_ZTIM1AFvvE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSM1AFvvE> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTIFvvE> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1A> : !cir.ptr<!u8i>}>

// LLVM-DAG: @_ZTSM1AFvvE = linkonce_odr global [7 x i8] c"M1AFvvE"
// LLVM-DAG: @_ZTSFvvE = linkonce_odr global [4 x i8] c"FvvE"
// LLVM-DAG: @_ZTIFvvE = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 16), ptr @_ZTSFvvE }
// LLVM-DAG: @_ZTIM1AFvvE = constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 16), ptr @_ZTSM1AFvvE, i32 0, ptr @_ZTIFvvE, ptr @_ZTI1A }
// LLVM-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIM1AFvvE, ptr null)

// OGCG-DAG: @_ZTSM1AFvvE = linkonce_odr constant [8 x i8] c"M1AFvvE\00", comdat
// OGCG-DAG: @_ZTSFvvE = linkonce_odr constant [5 x i8] c"FvvE\00", comdat
// OGCG-DAG: @_ZTIFvvE = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 2), ptr @_ZTSFvvE }, comdat
// OGCG-DAG: @_ZTIM1AFvvE = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2), ptr @_ZTSM1AFvvE, i32 0, ptr @_ZTIFvvE, ptr @_ZTI1A }, comdat
// OGCG-DAG: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIM1AFvvE, ptr null)
