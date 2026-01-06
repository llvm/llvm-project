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

void unused_pointer_to_member_func(void (Foo::*func)(int)) {
}

// CIR-BEFORE: cir.func {{.*}} @_Z29unused_pointer_to_member_funcM3FooFviE(%[[ARG:.*]]: !cir.method<!cir.func<(!s32i)> in !rec_Foo>)
// CIR-BEFORE:   %[[FUNC:.*]] = cir.alloca !cir.method<!cir.func<(!s32i)> in !rec_Foo>, !cir.ptr<!cir.method<!cir.func<(!s32i)> in !rec_Foo>>, ["func", init]

// CIR-AFTER: !rec_anon_struct = !cir.record<struct  {!s64i, !s64i}>
// CIR-AFTER: cir.func {{.*}} @_Z29unused_pointer_to_member_funcM3FooFviE(%[[ARG:.*]]: !rec_anon_struct {{.*}})
// CIR-AFTER    %[[FUNC:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["func", init]

// NOTE: The difference between LLVM and OGCG are due to the lack of calling convention handling in CIR.

// LLVM: define {{.*}} void @_Z29unused_pointer_to_member_funcM3FooFviE({ i64, i64 } %[[ARG:.*]])
// LLVM:   %[[FUNC:.*]] = alloca { i64, i64 }
// LLVM:   store { i64, i64 } %[[ARG]], ptr %[[FUNC]]

// OGCG: define {{.*}} void @_Z29unused_pointer_to_member_funcM3FooFviE(i64 %[[FUNC_COERCE0:.*]], i64 %[[FUNC_COERCE1:.*]])
// OGCG:   %[[FUNC:.*]] = alloca { i64, i64 }
// OGCG:   %[[FUNC_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   %[[FUNC_0:.*]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[FUNC]], i32 0, i32 0
// OGCG:   store i64 %[[FUNC_COERCE0]], ptr %[[FUNC_0]]
// OGCG:   %[[FUNC_1:.*]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[FUNC]], i32 0, i32 1
// OGCG:   store i64 %[[FUNC_COERCE1]], ptr %[[FUNC_1]]
// OGCG:   %[[FUNC1:.*]] = load { i64, i64 }, ptr %[[FUNC]]
// OGCG:   store { i64, i64 } %[[FUNC1]], ptr %[[FUNC_ADDR]]
