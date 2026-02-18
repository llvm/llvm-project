// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct A {
  A() = default;
  A(int); // This constructor triggers the null base class initialization.
};

struct B : A {
};

void test_empty_base_null_init() {
  B{};
}

// CIR: cir.func {{.*}} @_Z25test_empty_base_null_initv()
// CIR-NEXT:   %[[B_ADDR:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["agg.tmp.ensured"]
// CIR-NEXT:   %[[A_ADDR:.*]] = cir.base_class_addr %[[B_ADDR]] : !cir.ptr<!rec_B> nonnull [0] -> !cir.ptr<!rec_A>

// LLVM: define{{.*}} @_Z25test_empty_base_null_initv()
// LLVM-NEXT:   %[[B:.*]] = alloca %struct.B
// LLVM-NEXT:   ret void

// OGCG: define{{.*}} @_Z25test_empty_base_null_initv()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[B:.*]] = alloca %struct.B
// OGCG-NEXT:   ret void


struct C {
  int c;
  C() = default;
  C(int); // This constructor triggers the null base class initialization.
};

struct D : C {
};

void test_non_empty_base_null_init() {
  D{};
}

// CIR: cir.func {{.*}} @_Z29test_non_empty_base_null_initv()
// CIR:   %[[TMP:.*]] = cir.alloca !rec_D, !cir.ptr<!rec_D>, ["agg.tmp.ensured"]
// CIR:   %[[BASE:.*]] = cir.base_class_addr %[[TMP]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_C>
// CIR:   %[[ZERO:.*]] = cir.const #cir.const_record<{#cir.int<0> : !s32i}> : !rec_C
// CIR:   cir.store{{.*}} %[[ZERO]], %[[BASE]]

// LLVM: define{{.*}} void @_Z29test_non_empty_base_null_initv()
// LLVM:   %[[TMP:.*]] = alloca %struct.D
// LLVM:   store %struct.C zeroinitializer, ptr %[[TMP]]

// OGCG: define {{.*}} void @_Z29test_non_empty_base_null_initv()
// OGCG:   %[[TMP:.*]] = alloca %struct.D
// OGCG:   %[[BASE:.*]] = getelementptr inbounds i8, ptr %[[TMP]], i64 0
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[BASE]], i8 0, i64 4, i1 false)

struct E {
  int e;
};

struct F : E {
  F() = default;
  F(int);
};

struct G : F {
};

void test_base_chain_null_init() {
  G{};
}

// CIR: cir.func {{.*}} @_Z25test_base_chain_null_initv()
// CIR:   %[[TMP:.*]] = cir.alloca !rec_G, !cir.ptr<!rec_G>, ["agg.tmp.ensured"]
// CIR:   %[[BASE:.*]] = cir.base_class_addr %[[TMP]] : !cir.ptr<!rec_G> nonnull [0] -> !cir.ptr<!rec_F>
// CIR:   %[[ZERO:.*]] = cir.const #cir.const_record<{#cir.zero : !rec_E}> : !rec_F
// CIR:   cir.store{{.*}} %[[ZERO]], %[[BASE]]

// LLVM: define{{.*}} void @_Z25test_base_chain_null_initv()
// LLVM:   %[[TMP:.*]] = alloca %struct.G
// LLVM:   store %struct.F zeroinitializer, ptr %[[TMP]]

// OGCG: define {{.*}} void @_Z25test_base_chain_null_initv()
// OGCG:   %[[TMP:.*]] = alloca %struct.G
// OGCG:   %[[BASE:.*]] = getelementptr inbounds i8, ptr %[[TMP]], i64 0
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[BASE]], i8 0, i64 4, i1 false)
