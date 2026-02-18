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
// CIR-BEFORE:   %{{.*}} = cir.cast member_ptr_to_bool %{{.*}} : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo> -> !cir.bool

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
// CIR-BEFORE:   %{{.*}} = cir.cast bitcast %{{.*}} : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo> -> !cir.method<!cir.func<(!cir.ptr<!rec_Bar>)> in !rec_Bar>

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

struct Base1 {
  int x;
  virtual void m1(int);
};

struct Base2 {
  int y;
  virtual void m2(int);
};

struct Derived : Base1, Base2 {
  virtual void m3(int);
};

using Base1MemFunc = void (Base1::*)(int);
using Base2MemFunc = void (Base2::*)(int);
using DerivedMemFunc = void (Derived::*)(int);

DerivedMemFunc base_to_derived_zero_offset(Base1MemFunc ptr) {
  return static_cast<DerivedMemFunc>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z27base_to_derived_zero_offsetM5Base1FviE
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Base1>, !s32i)> in !rec_Base1>>, !cir.method<!cir.func<(!cir.ptr<!rec_Base1>, !s32i)> in !rec_Base1>
// CIR-BEFORE:   %{{.*}} = cir.derived_method %[[PTR]][0] : !cir.method<!cir.func<(!cir.ptr<!rec_Base1>, !s32i)> in !rec_Base1> -> !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>

// CIR-AFTER: cir.func {{.*}} @_Z27base_to_derived_zero_offsetM5Base1FviE
// CIR-AFTER:   %[[PTR:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["ptr", init]
// CIR-AFTER:   %[[RET:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   cir.store %{{.*}}, %[[PTR]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[TMP:.*]] = cir.load{{.*}} %[[PTR]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.store %[[TMP]], %[[RET]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET_VAL:.*]] = cir.load %[[RET]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.return %[[RET_VAL]] : !rec_anon_struct

// LLVM: define {{.*}} { i64, i64 } @_Z27base_to_derived_zero_offsetM5Base1FviE
// LLVM:   %[[ARG_ADDR:.*]] = alloca { i64, i64 }
// LLVM:   %[[RET_ADDR:.*]] = alloca { i64, i64 }
// LLVM:   store { i64, i64 } %{{.*}}, ptr %[[ARG_ADDR]]
// LLVM:   %[[TMP:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// LLVM:   store { i64, i64 } %[[TMP]], ptr %[[RET_ADDR]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RET_ADDR]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG: define {{.*}} { i64, i64 } @_Z27base_to_derived_zero_offsetM5Base1FviE
// OGCG:   %[[ARG_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   store { i64, i64 } %{{.*}}, ptr %[[ARG_ADDR]]
// OGCG:   %[[RET:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// OGCG:   ret { i64, i64 } %[[RET]]

DerivedMemFunc base_to_derived(Base2MemFunc ptr) {
  return static_cast<DerivedMemFunc>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z15base_to_derivedM5Base2FviE
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Base2>, !s32i)> in !rec_Base2>>, !cir.method<!cir.func<(!cir.ptr<!rec_Base2>, !s32i)> in !rec_Base2>
// CIR-BEFORE:   %{{.*}} = cir.derived_method %[[PTR]][16] : !cir.method<!cir.func<(!cir.ptr<!rec_Base2>, !s32i)> in !rec_Base2> -> !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>

// CIR-AFTER: cir.func {{.*}} @_Z15base_to_derivedM5Base2FviE
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[OFFSET:.*]] = cir.extract_member %[[PTR]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[OFFSET_ADJ:.*]] = cir.const #cir.int<16> : !s64i
// CIR-AFTER:   %[[BINOP_KIND:.*]] = cir.binop(add, %[[OFFSET]], %[[OFFSET_ADJ]]) nsw : !s64i
// CIR-AFTER:   %{{.*}} = cir.insert_member %[[PTR]][1], %[[BINOP_KIND]] : !rec_anon_struct, !s64i

// LLVM: define {{.*}} { i64, i64 } @_Z15base_to_derivedM5Base2FviE
// LLVM:   %[[ARG:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   %[[ADJ:.*]] = extractvalue { i64, i64 } %[[ARG]], 1
// LLVM:   %[[ADJ_ADJ:.*]] = add nsw i64 %[[ADJ]], 16
// LLVM:   %{{.*}} = insertvalue { i64, i64 } %[[ARG]], i64 %[[ADJ_ADJ]], 1

// OGCG: define {{.*}} { i64, i64 } @_Z15base_to_derivedM5Base2FviE
// OGCG:   %[[ARG:.*]] = load { i64, i64 }, ptr %{{.*}}
// OGCG:   store { i64, i64 } %[[ARG]], ptr %[[ARG_ADDR:.*]]
// OGCG:   %[[ARG1:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// OGCG:   %[[ADJ:.*]] = extractvalue { i64, i64 } %[[ARG1]], 1
// OGCG:   %[[ADJ_ADJ:.*]] = add nsw i64 %[[ADJ]], 16
// OGCG:   %{{.*}} = insertvalue { i64, i64 } %[[ARG1]], i64 %[[ADJ_ADJ]], 1

Base1MemFunc derived_to_base_zero_offset(DerivedMemFunc ptr) {
  return static_cast<Base1MemFunc>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z27derived_to_base_zero_offsetM7DerivedFviE
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>>, !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>
// CIR-BEFORE:   %{{.*}} = cir.base_method %[[PTR]][0] : !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived> -> !cir.method<!cir.func<(!cir.ptr<!rec_Base1>, !s32i)> in !rec_Base1>

// CIR-AFTER: cir.func {{.*}} @_Z27derived_to_base_zero_offsetM7DerivedFviE
// CIR-AFTER:   %[[PTR:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["ptr", init]
// CIR-AFTER:   %[[RET:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   cir.store %{{.*}}, %[[PTR]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[TMP:.*]] = cir.load{{.*}} %[[PTR]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.store %[[TMP]], %[[RET]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET_VAL:.*]] = cir.load %[[RET]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   cir.return %[[RET_VAL]] : !rec_anon_struct

// LLVM: define {{.*}} { i64, i64 } @_Z27derived_to_base_zero_offsetM7DerivedFviE
// LLVM:   %[[ARG_ADDR:.*]] = alloca { i64, i64 }
// LLVM:   %[[RET_ADDR:.*]] = alloca { i64, i64 }
// LLVM:   store { i64, i64 } %{{.*}}, ptr %[[ARG_ADDR]]
// LLVM:   %[[TMP:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// LLVM:   store { i64, i64 } %[[TMP]], ptr %[[RET_ADDR]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RET_ADDR]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG: define {{.*}} { i64, i64 } @_Z27derived_to_base_zero_offsetM7DerivedFviE
// OGCG:   %[[ARG_ADDR:.*]] = alloca { i64, i64 }
// OGCG:   store { i64, i64 } %{{.*}}, ptr %[[ARG_ADDR]]
// OGCG:   %[[RET:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// OGCG:   ret { i64, i64 } %[[RET]]

Base2MemFunc derived_to_base(DerivedMemFunc ptr) {
  return static_cast<Base2MemFunc>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z15derived_to_baseM7DerivedFviE
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>>, !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived>
// CIR-BEFORE:   %{{.*}} = cir.base_method %[[PTR]][16] : !cir.method<!cir.func<(!cir.ptr<!rec_Derived>, !s32i)> in !rec_Derived> -> !cir.method<!cir.func<(!cir.ptr<!rec_Base2>, !s32i)> in !rec_Base2>

// CIR-AFTER: cir.func {{.*}} @_Z15derived_to_baseM7DerivedFviE
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:   %[[OFFSET:.*]] = cir.extract_member %[[PTR]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER:   %[[OFFSET_ADJ:.*]] = cir.const #cir.int<16> : !s64i
// CIR-AFTER:   %[[BINOP_KIND:.*]] = cir.binop(sub, %[[OFFSET]], %[[OFFSET_ADJ]]) nsw : !s64i
// CIR-AFTER:   %{{.*}} = cir.insert_member %[[PTR]][1], %[[BINOP_KIND]] : !rec_anon_struct, !s64i

// LLVM: define {{.*}} { i64, i64 } @_Z15derived_to_baseM7DerivedFviE
// LLVM:   %[[ARG:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:   %[[ADJ:.*]] = extractvalue { i64, i64 } %[[ARG]], 1
// LLVM:   %[[ADJ_ADJ:.*]] = sub nsw i64 %[[ADJ]], 16
// LLVM:   %{{.*}} = insertvalue { i64, i64 } %[[ARG]], i64 %[[ADJ_ADJ]], 1

// OGCG: define {{.*}} { i64, i64 } @_Z15derived_to_baseM7DerivedFviE
// OGCG:   %[[ARG:.*]] = load { i64, i64 }, ptr %{{.*}}
// OGCG:   store { i64, i64 } %[[ARG]], ptr %[[ARG_ADDR:.*]]
// OGCG:   %[[ARG1:.*]] = load { i64, i64 }, ptr %[[ARG_ADDR]]
// OGCG:   %[[ADJ:.*]] = extractvalue { i64, i64 } %[[ARG1]], 1
// OGCG:   %[[ADJ_ADJ:.*]] = sub nsw i64 %[[ADJ]], 16
// OGCG:   %{{.*}} = insertvalue { i64, i64 } %[[ARG1]], i64 %[[ADJ_ADJ]], 1
