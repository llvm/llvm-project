// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-OGCG

struct Struct {
  virtual void f1() = 0;
  virtual void f2() = delete;
  virtual void f3();
};

void Struct::f3(){}


// CIR: cir.global "private" external @_ZTV6Struct = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI6Struct> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_deleted_virtual> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Struct2f3Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>}>
// LLVM: @_ZTV6Struct = {{.*}}{ [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI6Struct, ptr @__cxa_pure_virtual, ptr @__cxa_deleted_virtual, ptr @_ZN6Struct2f3Ev] }

// CIR: cir.global constant external @_ZTI6Struct = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS6Struct> : !cir.ptr<!u8i>}>
// LLVM-CIR: @_ZTI6Struct = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS6Struct }
// LLVM-OGCG: @_ZTI6Struct = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS6Struct }
//
// CIR: cir.func private dso_local @__cxa_pure_virtual()
// LLVM: declare {{.*}}void @__cxa_pure_virtual()
//
// CIR: cir.func private dso_local @__cxa_deleted_virtual()
// LLVM: declare {{.*}}void @__cxa_deleted_virtual()
