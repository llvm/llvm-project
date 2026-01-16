// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

auto make_non_virtual() -> void (Foo::*)(int) {
  return &Foo::m1;
}

// CIR-BEFORE: cir.func {{.*}} @_Z16make_non_virtualv() -> !cir.method<!cir.func<(!s32i)> in !rec_Foo>
// CIR-BEFORE:   %[[RETVAL:.*]] = cir.alloca !cir.method<!cir.func<(!s32i)> in !rec_Foo>, !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>, ["__retval"]
// CIR-BEFORE:   %[[METHOD_PTR:.*]] = cir.const #cir.method<@_ZN3Foo2m1Ei> : !cir.method<!cir.func<(!s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[METHOD_PTR]], %[[RETVAL]]
// CIR-BEFORE:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-BEFORE:   cir.return %[[RET]] : !cir.method<!cir.func<(!s32i)> in !rec_Foo>

// CIR-AFTER: cir.func {{.*}} @_Z16make_non_virtualv() -> !rec_anon_struct {
// CIR-AFTER:   %[[RETVAL:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   %[[METHOD_PTR:.*]] = cir.const #cir.const_record<{#cir.global_view<@_ZN3Foo2m1Ei> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// CIR-AFTER:   cir.store %[[METHOD_PTR]], %[[RETVAL]]
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-AFTER:   cir.return %[[RET]] : !rec_anon_struct

// LLVM: define {{.*}} { i64, i64 } @_Z16make_non_virtualv()
// LLVM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// LLVM:   store { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }, ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RETVAL]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG: define {{.*}} { i64, i64 } @_Z16make_non_virtualv()
// OGCG:   ret { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }
