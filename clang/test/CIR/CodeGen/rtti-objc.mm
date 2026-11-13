// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -x objective-c++
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll -x objective-c++
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll -x objective-c++
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// RTTI emission of ObjCObject for Objective C builtin types via typeid. Type operands were
// used to avoid using convertType(ObjCObjectPointerType) as it is NYI in
// CIRGenTypes.

namespace std { class type_info; }

void f() {
  const std::type_info &t1 = typeid(id);
  const std::type_info &t2 = typeid(Class);
}

// CIR-DAG: cir.global {{.*}} @_ZTS11objc_object = #cir.const_array<"11objc_object" : !cir.array<!s8i x 13>, trailing_zeros> : !cir.array<!s8i x 14>
// CIR-DAG: cir.global {{.*}} @_ZTI11objc_object = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS11objc_object> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.global {{.*}} @_ZTSP11objc_object = #cir.const_array<"P11objc_object" : !cir.array<!s8i x 14>, trailing_zeros> : !cir.array<!s8i x 15>
// CIR-DAG: cir.global {{.*}} @_ZTIP11objc_object = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv119__pointer_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSP11objc_object> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTI11objc_object> : !cir.ptr<!u8i>}>

// CIR-DAG: cir.global {{.*}} @_ZTS10objc_class = #cir.const_array<"10objc_class" : !cir.array<!s8i x 12>, trailing_zeros> : !cir.array<!s8i x 13>
// CIR-DAG: cir.global {{.*}} @_ZTI10objc_class = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS10objc_class> : !cir.ptr<!u8i>}>
// CIR-DAG: cir.global {{.*}} @_ZTSP10objc_class = #cir.const_array<"P10objc_class" : !cir.array<!s8i x 13>, trailing_zeros> : !cir.array<!s8i x 14>
// CIR-DAG: cir.global {{.*}} @_ZTIP10objc_class = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv119__pointer_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSP10objc_class> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.global_view<@_ZTI10objc_class> : !cir.ptr<!u8i>}>

// LLVM-DAG: @_ZTS11objc_object = linkonce_odr global [14 x i8] c"11objc_object\00", comdat
// LLVM-DAG: @_ZTI11objc_object = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS11objc_object }, comdat
// LLVM-DAG: @_ZTSP11objc_object = linkonce_odr global [15 x i8] c"P11objc_object\00", comdat
// LLVM-DAG: @_ZTIP11objc_object = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP11objc_object, i32 0, ptr @_ZTI11objc_object }, comdat

// LLVM-DAG: @_ZTS10objc_class = linkonce_odr global [13 x i8] c"10objc_class\00", comdat
// LLVM-DAG: @_ZTI10objc_class = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS10objc_class }, comdat
// LLVM-DAG: @_ZTSP10objc_class = linkonce_odr global [14 x i8] c"P10objc_class\00", comdat
// LLVM-DAG: @_ZTIP10objc_class = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP10objc_class, i32 0, ptr @_ZTI10objc_class }, comdat

// OGCG-DAG: @_ZTS11objc_object = linkonce_odr constant [14 x i8] c"11objc_object\00", comdat
// OGCG-DAG: @_ZTI11objc_object = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS11objc_object }, comdat
// OGCG-DAG: @_ZTSP11objc_object = linkonce_odr constant [15 x i8] c"P11objc_object\00", comdat
// OGCG-DAG: @_ZTIP11objc_object = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSP11objc_object, i32 0, ptr @_ZTI11objc_object }, comdat

// OGCG-DAG: @_ZTS10objc_class = linkonce_odr constant [13 x i8] c"10objc_class\00", comdat
// OGCG-DAG: @_ZTI10objc_class = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS10objc_class }, comdat
// OGCG-DAG: @_ZTSP10objc_class = linkonce_odr constant [14 x i8] c"P10objc_class\00", comdat
// OGCG-DAG: @_ZTIP10objc_class = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSP10objc_class, i32 0, ptr @_ZTI10objc_class }, comdat
