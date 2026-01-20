// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

struct Bar {
  void m4();
};

bool memfunc_to_bool(void (Foo::*func)(int)) {
  return func;
}

// CIR-BEFORE: cir.func {{.*}} @_Z15memfunc_to_boolM3FooFviE
// CIR-BEFORE:   %{{.*}} = cir.cast member_ptr_to_bool %{{.*}} : !cir.method<!cir.func<(!s32i)> in !rec_Foo> -> !cir.bool

// CIR-AFTER: cir.func {{.*}} @_Z15memfunc_to_boolM3FooFviE
// CIR-AFTER:   %[[FUNC:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[NULL_VAL:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER:   %[[FUNC_PTR:.*]] = cir.extract_member %[[FUNC]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[BOOL_VAL:.*]] = cir.cmp(ne, %[[FUNC_PTR]], %[[NULL_VAL]]) : !s64i, !cir.bool

// LLVM: define {{.*}} i1 @_Z15memfunc_to_boolM3FooFviE
// LLVM:   %[[FUNC:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   %[[FUNC_PTR:.*]] = extractvalue { i64, i64 } %[[FUNC]], 0
// LLVM:   %{{.*}} = icmp ne i64 %[[FUNC_PTR]], 0

// Note: OGCG uses an extra temporary for the function argument because it
//       composes it from coerced arguments. We'll do that in CIR too after
//       calling convention lowering is implemented.

// OGCG: define {{.*}} i1 @_Z15memfunc_to_boolM3FooFviE
// OGCG:   %[[FUNC_TMP:.*]] = load { i64, i64 }, ptr %{{.*}}
// OGCG:   store { i64, i64 } %[[FUNC_TMP]], ptr %[[FUNC_ADDR:.*]]
// OGCG:   %[[FUNC:.*]] = load { i64, i64 }, ptr %[[FUNC_ADDR]]
// OGCG:   %[[FUNC_PTR:.*]] = extractvalue { i64, i64 } %[[FUNC]], 0
// OGCG:   %{{.*}} = icmp ne i64 %[[FUNC_PTR]], 0

auto memfunc_reinterpret(void (Foo::*func)(int)) -> void (Bar::*)() {
  return reinterpret_cast<void (Bar::*)()>(func);
}

// CIR-BEFORE: cir.func {{.*}} @_Z19memfunc_reinterpretM3FooFviE
// CIR-BEFORE:   %{{.*}} = cir.cast bitcast %{{.*}} : !cir.method<!cir.func<(!s32i)> in !rec_Foo> -> !cir.method<!cir.func<()> in !rec_Bar>

// CIR-AFTER: cir.func {{.*}} @_Z19memfunc_reinterpretM3FooFviE
// CIR-AFTER:   %[[FUNC:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.store %[[FUNC]], %[[RET_ADDR:.*]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET:.*]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.return %[[RET]] : !rec_anon_struct

// LLVM: define {{.*}} { i64, i64 } @_Z19memfunc_reinterpretM3FooFviE
// LLVM:   %[[FUNC:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   store { i64, i64 } %[[FUNC]], ptr %[[RET_ADDR:.*]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RET_ADDR]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG: define {{.*}} { i64, i64 } @_Z19memfunc_reinterpretM3FooFviE
// OGCG:   %[[FUNC:.*]] = load { i64, i64 }, ptr %{{.*}}
// OGCG:   store { i64, i64 } %[[FUNC]], ptr %[[RET_ADDR:.*]]
// OGCG:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RET_ADDR]]
// OGCG:   ret { i64, i64 } %[[RET]]
