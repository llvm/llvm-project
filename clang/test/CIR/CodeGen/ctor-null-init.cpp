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
