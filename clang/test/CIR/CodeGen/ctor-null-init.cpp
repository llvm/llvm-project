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
// CIR-NEXT:   %[[A_ADDR:.*]] = cir.base_class_addr nonnull %[[B_ADDR]] [0] : !cir.ptr<!rec_B> -> !cir.ptr<!rec_A>

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
// CIR:   %[[BASE:.*]] = cir.base_class_addr nonnull %[[TMP]] [0] : !cir.ptr<!rec_D> -> !cir.ptr<!rec_C>
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
// CIR:   %[[BASE:.*]] = cir.base_class_addr nonnull %[[TMP]] [0] : !cir.ptr<!rec_G> -> !cir.ptr<!rec_F>
// CIR:   %[[ZERO:.*]] = cir.const #cir.const_record<{#cir.zero : !rec_E}> : !rec_F
// CIR:   cir.store{{.*}} %[[ZERO]], %[[BASE]]

// LLVM: define{{.*}} void @_Z25test_base_chain_null_initv()
// LLVM:   %[[TMP:.*]] = alloca %struct.G
// LLVM:   store %struct.F zeroinitializer, ptr %[[TMP]]

// OGCG: define {{.*}} void @_Z25test_base_chain_null_initv()
// OGCG:   %[[TMP:.*]] = alloca %struct.G
// OGCG:   %[[BASE:.*]] = getelementptr inbounds i8, ptr %[[TMP]], i64 0
// OGCG:   call void @llvm.memset.p0.i64(ptr{{.*}} %[[BASE]], i8 0, i64 4, i1 false)

struct VBase {
  virtual ~VBase();
};

struct VDerived : VBase {
  VDerived();
};
VDerived::VDerived() : VBase() {}

// OGCG-LABEL: define {{.*}}@_ZN8VDerivedC2Ev
// OGCG: %[[ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS:.*]],
// OGCG: call void @llvm.memset.p0.i64(ptr align 8 %[[ADDR]], i8 0, i64 8, i1 false)
// OGCG: call void @_ZN5VBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %[[THIS]])
// OGCG: store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV8VDerived, i32 0, i32 0, i32 2), ptr %[[THIS]], align 8

// CIR-LABEL: cir.func {{.*}}@_ZN5VBaseC2Ev
// CIR: cir.vtable.address_point(@_ZTV5VBase
// CIR: cir.vtable.get_vptr

// LLVM-LABEL: define {{.*}}@_ZN5VBaseC2Ev
// LLVM: store ptr getelementptr inbounds nuw (i8, ptr @_ZTV5VBase, i64 16), ptr %
// OGCG-LABEL: define {{.*}}@_ZN5VBaseC2Ev
// OGCG: store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV5VBase, i32 0, i32 0, i32 2), ptr %

// CIR-LABEL: cir.func {{.*}}@_ZN8VDerivedC2Ev
// CIR: %[[BASE:.*]] = cir.base_class_addr {{.*}} : !cir.ptr<!rec_VDerived> nonnull [0] -> !cir.ptr<!rec_VBase>
// CIR: %[[ZERO:.*]] = cir.const #cir.const_record<{#cir.zero : !cir.vptr}> : !rec_VBase 
// CIR: cir.store align(8) %[[ZERO]], %[[BASE]] : !rec_VBase, !cir.ptr<!rec_VBase>
// CIR: cir.call @_ZN5VBaseC2Ev(%[[BASE]])
// CIR: cir.vtable.address_point(@_ZTV8VDerived
// CIR:  %5 = cir.vtable.get_vptr

// LLVM-LABEL: define {{.*}}@_ZN8VDerivedC2Ev
// LLVM: store %struct.VBase zeroinitializer, ptr %[[ADDR:.*]], align 8
// LLVM: call void @_ZN5VBaseC2Ev(ptr {{.*}}%[[ADDR]])
// LLVM: store ptr getelementptr inbounds nuw (i8, ptr @_ZTV8VDerived, i64 16), ptr %[[ADDR]]
