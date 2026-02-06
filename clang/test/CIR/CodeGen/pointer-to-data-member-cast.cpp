// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct Base1 {
  int base1_data;
};

struct Base2 {
  int base2_data;
};

struct Derived : Base1, Base2 {
  int derived_data;
};

auto base_to_derived(int Base2::*ptr) -> int Derived::* {
  return ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z15base_to_derivedM5Base2i
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[RET:.*]] = cir.derived_data_member %[[PTR]][4] : !cir.data_member<!s32i in !rec_Base2> -> !cir.data_member<!s32i in !rec_Derived>

// CIR-AFTER: cir.func {{.*}} @_Z15base_to_derivedM5Base2i
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %[[NULL_VALUE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR-AFTER:   %[[IS_NULL:.*]] = cir.cmp(eq, %[[PTR]], %[[NULL_VALUE]])
// CIR-AFTER:   %[[OFFSET_VALUE:.*]] = cir.const #cir.int<4> : !s64i
// CIR-AFTER:   %[[BINOP_KIND:.*]] = cir.binop(add, %[[PTR]], %[[OFFSET_VALUE]]) nsw : !s64i
// CIR-AFTER:   %[[SELECT:.*]] = cir.select if %[[IS_NULL]] then %[[PTR]] else %[[BINOP_KIND]]

// LLVM: define {{.*}} i64 @_Z15base_to_derivedM5Base2i
// LLVM:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// LLVM:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// LLVM:   %[[DERIVED:.*]] = add nsw i64 %[[PTR]], 4
// LLVM:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[DERIVED]]

// OGCG: define {{.*}} i64 @_Z15base_to_derivedM5Base2i
// OGCG:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// OGCG:   %[[DERIVED:.*]] = add nsw i64 %[[PTR]], 4
// OGCG:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// OGCG:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[DERIVED]]

auto derived_to_base(int Derived::*ptr) -> int Base2::* {
  return static_cast<int Base2::*>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z15derived_to_baseM7Derivedi
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[RET:.*]] = cir.base_data_member %[[PTR]][4] : !cir.data_member<!s32i in !rec_Derived> -> !cir.data_member<!s32i in !rec_Base2>

// CIR-AFTER: cir.func {{.*}} @_Z15derived_to_baseM7Derivedi
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %[[NULL_VALUE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR-AFTER:   %[[IS_NULL:.*]] = cir.cmp(eq, %[[PTR]], %[[NULL_VALUE]])
// CIR-AFTER:   %[[OFFSET_VALUE:.*]] = cir.const #cir.int<4> : !s64i
// CIR-AFTER:   %[[BINOP_KIND:.*]] = cir.binop(sub, %[[PTR]], %[[OFFSET_VALUE]]) nsw : !s64i
// CIR-AFTER:   %[[SELECT:.*]] = cir.select if %[[IS_NULL]] then %[[PTR]] else %[[BINOP_KIND]]

// LLVM: define {{.*}} i64 @_Z15derived_to_baseM7Derivedi
// LLVM:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// LLVM:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// LLVM:   %[[BASE:.*]] = sub nsw i64 %[[PTR]], 4
// LLVM:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[BASE]]

// OGCG: define {{.*}} i64 @_Z15derived_to_baseM7Derivedi
// OGCG:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// OGCG:   %[[BASE:.*]] = sub nsw i64 %[[PTR]], 4
// OGCG:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// OGCG:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[BASE]]

auto base_to_derived_zero_offset(int Base1::*ptr) -> int Derived::* {
  return ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z27base_to_derived_zero_offsetM5Base1i
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[RET:.*]] = cir.derived_data_member %[[PTR]][0] : !cir.data_member<!s32i in !rec_Base1> -> !cir.data_member<!s32i in !rec_Derived>

// CIR-AFTER: cir.func {{.*}} @_Z27base_to_derived_zero_offsetM5Base1i
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   cir.store %[[PTR]], %{{.*}} : !s64i, !cir.ptr<!s64i>

// No LLVM instructions emitted for performing a zero-offset cast.

// LLVM:      define {{.*}} i64 @_Z27base_to_derived_zero_offsetM5Base1i
// LLVM-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca i64
// LLVM-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   %[[TEMP:.*]] = load i64, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   store i64 %[[TEMP]], ptr %[[RETVAL]]
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM-NEXT:   ret i64 %[[RET]]

// OGCG:      define {{.*}} i64 @_Z27base_to_derived_zero_offsetM5Base1i
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// OGCG-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   ret i64 %[[RET]]

auto derived_to_base_zero_offset(int Derived::*ptr) -> int Base1::* {
  return static_cast<int Base1::*>(ptr);
}

// CIR-BEFORE: cir.func {{.*}} @_Z27derived_to_base_zero_offsetM7Derivedi
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[RET:.*]] = cir.base_data_member %[[PTR]][0] : !cir.data_member<!s32i in !rec_Derived> -> !cir.data_member<!s32i in !rec_Base1>

// CIR-AFTER: cir.func {{.*}} @_Z27derived_to_base_zero_offsetM7Derivedi
// CIR-AFTER:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   cir.store %[[PTR]], %{{.*}} : !s64i, !cir.ptr<!s64i>

// No LLVM instructions emitted for performing a zero-offset cast.

// LLVM:      define {{.*}} i64 @_Z27derived_to_base_zero_offsetM7Derivedi
// LLVM-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca i64
// LLVM-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   %[[TEMP:.*]] = load i64, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   store i64 %[[TEMP]], ptr %[[RETVAL]]
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM-NEXT:   ret i64 %[[RET]]

// OGCG:      define {{.*}} i64 @_Z27derived_to_base_zero_offsetM7Derivedi
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// OGCG-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   ret i64 %[[RET]]

struct Foo {
  int a;
};

struct Bar {
  int a;
};

bool to_bool(int Foo::*x) {
  return x;
}

// CIR-BEFORE: cir.func {{.*}} @_Z7to_boolM3Fooi
// CIR-BEFORE:   %[[X:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE:   %{{.*}} = cir.cast member_ptr_to_bool %[[X]] : !cir.data_member<!s32i in !rec_Foo> -> !cir.bool

// CIR-AFTER: cir.func {{.*}} @_Z7to_boolM3Fooi
// CIR-AFTER:   %[[NULL_VAL:.*]] = cir.const #cir.int<-1> : !s64i
// CIR-AFTER:   %[[BOOL_VAL:.*]] = cir.cmp(ne, %{{.*}}, %[[NULL_VAL]]) : !s64i, !cir.bool

// LLVM: define {{.*}} i1 @_Z7to_boolM3Fooi
// LLVM:   %[[X:.*]] = load i64, ptr %{{.*}}
// LLVM:   %[[IS_NULL:.*]] = icmp ne i64 %[[X]], -1

// OGCG: define {{.*}} i1 @_Z7to_boolM3Fooi
// OGCG:   %[[X:.*]] = load i64, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp ne i64 %[[X]], -1

auto bitcast(int Foo::*x) {
  return reinterpret_cast<int Bar::*>(x);
}

// CIR-BEFORE: cir.func {{.*}} @_Z7bitcastM3Fooi
// CIR-BEFORE:   %[[X:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE:   %{{.*}} = cir.cast bitcast %[[X]] : !cir.data_member<!s32i in !rec_Foo> -> !cir.data_member<!s32i in !rec_Bar>

// CIR-AFTER: cir.func {{.*}} @_Z7bitcastM3Fooi(%[[ARG0:.*]]: !s64i
// CIR-AFTER:   %[[X_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["x", init] {alignment = 8 : i64}
// CIR-AFTER:   %[[RET_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"] {alignment = 8 : i64}
// CIR-AFTER:   cir.store %[[ARG0]], %[[X_ADDR]]
// CIR-AFTER:   %[[X:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR-AFTER:   cir.store %[[X]], %[[RET_ADDR]]
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RET_ADDR]]
// CIR-AFTER:   cir.return %[[RET]] : !s64i

// LLVM: define {{.*}} i64 @_Z7bitcastM3Fooi
// LLVM:   %[[X:.*]] = load i64, ptr %{{.*}}
// LLVM:   store i64 %[[X]], ptr %[[RET_ADDR:.*]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RET_ADDR:.*]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z7bitcastM3Fooi
// OGCG:   %[[X:.*]] = load i64, ptr %{{.*}}
// OGCG:   ret i64 %[[X]]
