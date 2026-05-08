// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct HasVal {
  int x = 5;
};
struct WithCtor {
  consteval WithCtor(int a, long b) : x(a*a), y(a*b) {}
  int x;
  long y;
  HasVal a;
};

extern "C" void construct() {
  WithCtor c(2,5);
}

// CIR-LABEL: construct()
// CIR-NEXT: %[[WC_ALLOCA:.*]] = cir.alloca !rec_WithCtor
// CIR-NEXT: %[[CONST_VAL:.*]] = cir.const #cir.const_record<{#cir.int<4> : !s32i, #cir.int<10> : !s64i, #cir.const_record<{#cir.int<5> : !s32i}> : !rec_HasVal}>
// CIR-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[WC_ALLOCA]]
// CIR-NEXT: cir.store{{.*}}%[[CONST_VAL]], %[[BITCAST]]

// LLVM-LABEL: construct()
// LLVM-NEXT: %[[WC_ALLOCA:.*]] = alloca %struct.WithCtor
// LLVM-NEXT: store { i32, i64, %struct.HasVal } { i32 4, i64 10, %struct.HasVal { i32 5 } }, ptr %[[WC_ALLOCA]]

// OGCG-LABEL: construct()
// OGCG: %[[WC_ALLOCA:.*]] = alloca %struct.WithCtor
// OGCG-NEXT: %[[X_GEP:.*]] = getelementptr{{.*}} { i32, i64, %struct.HasVal }, ptr %[[WC_ALLOCA]], i32 0, i32 0
// OGCG-NEXT: store i32 4, ptr %[[X_GEP]]
// OGCG-NEXT: %[[Y_GEP:.*]] = getelementptr{{.*}} { i32, i64, %struct.HasVal }, ptr %[[WC_ALLOCA]], i32 0, i32 1
// OGCG-NEXT: store i64 10, ptr %[[Y_GEP]]
// OGCG-NEXT: %[[A_GEP:.*]] = getelementptr{{.*}} { i32, i64, %struct.HasVal }, ptr %[[WC_ALLOCA]], i32 0, i32 2
// OGCG-NEXT: store %struct.HasVal { i32 5 }, ptr %[[A_GEP]]
