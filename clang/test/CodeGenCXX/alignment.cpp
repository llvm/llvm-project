// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin10 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOCOMPAT
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin10 -fclang-abi-compat=6.0 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V6COMPAT
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-scei-ps4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V6COMPAT

extern int int_source();
extern void int_sink(int x);

namespace test0 {
  struct A {
    int aField;
    int bField;
  };

  struct B {
    int onebit : 2;
    int twobit : 6;
    int intField;
  };

  struct __attribute__((packed, aligned(2))) C : A, B {
  };

  // These accesses should have alignment 4 because they're at offset 0
  // in a reference with an assumed alignment of 4.
  // CHECK-LABEL: @_ZN5test01aERNS_1BE
  void a(B &b) {
    // CHECK: [[CALL:%.*]] = call noundef i32 @_Z10int_sourcev()
    // CHECK: [[B_P:%.*]] = load [[B:%.*]]*, [[B]]**
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i8
    // CHECK: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = and i8 [[TRUNC]], 3
    // CHECK: [[T1:%.*]] = and i8 [[OLD_VALUE]], -4
    // CHECK: [[T2:%.*]] = or i8 [[T1]], [[T0]]
    // CHECK: store i8 [[T2]], i8* [[FIELD_P]], align 4
    b.onebit = int_source();

    // CHECK: [[B_P:%.*]] = load [[B]]*, [[B]]**
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = shl i8 [[VALUE]], 6
    // CHECK: [[T1:%.*]] = ashr i8 [[T0]], 6
    // CHECK: [[T2:%.*]] = sext i8 [[T1]] to i32
    // CHECK: call void @_Z8int_sinki(i32 noundef [[T2]])
    int_sink(b.onebit);
  }

  // These accesses should have alignment 2 because they're at offset 8
  // in a reference/pointer with an assumed alignment of 2.
  // CHECK-LABEL: @_ZN5test01bERNS_1CE
  void b(C &c) {
    // CHECK: [[CALL:%.*]] = call noundef i32 @_Z10int_sourcev()
    // CHECK: [[C_P:%.*]] = load [[C:%.*]]*, [[C]]**
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i8
    // CHECK-V6COMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = and i8 [[TRUNC]], 3
    // CHECK: [[T1:%.*]] = and i8 [[OLD_VALUE]], -4
    // CHECK: [[T2:%.*]] = or i8 [[T1]], [[T0]]
    // CHECK-V6COMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 4
    c.onebit = int_source();

    // CHECK: [[C_P:%.*]] = load [[C]]*, [[C]]**
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK-V6COMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = shl i8 [[VALUE]], 6
    // CHECK: [[T1:%.*]] = ashr i8 [[T0]], 6
    // CHECK: [[T2:%.*]] = sext i8 [[T1]] to i32
    // CHECK: call void @_Z8int_sinki(i32 noundef [[T2]])
    int_sink(c.onebit);
  }

  // CHECK-LABEL: @_ZN5test01cEPNS_1CE
  void c(C *c) {
    // CHECK: [[CALL:%.*]] = call noundef i32 @_Z10int_sourcev()
    // CHECK: [[C_P:%.*]] = load [[C]]*, [[C]]**
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i8
    // CHECK-V6COMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = and i8 [[TRUNC]], 3
    // CHECK: [[T1:%.*]] = and i8 [[OLD_VALUE]], -4
    // CHECK: [[T2:%.*]] = or i8 [[T1]], [[T0]]
    // CHECK-V6COMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 4
    c->onebit = int_source();

    // CHECK: [[C_P:%.*]] = load [[C:%.*]]*, [[C]]**
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B:%.*]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK-V6COMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = shl i8 [[VALUE]], 6
    // CHECK: [[T1:%.*]] = ashr i8 [[T0]], 6
    // CHECK: [[T2:%.*]] = sext i8 [[T1]] to i32
    // CHECK: call void @_Z8int_sinki(i32 noundef [[T2]])
    int_sink(c->onebit);
  }

  // These accesses should have alignment 2 because they're at offset 8
  // in an alignment-2 variable.
  // CHECK-LABEL: @_ZN5test01dEv
  void d() {
    // CHECK-V6COMPAT: [[C_P:%.*]] = alloca [[C:%.*]], align 2
    // CHECK-NOCOMPAT: [[C_P:%.*]] = alloca [[C:%.*]], align 4
    C c;

    // CHECK: [[CALL:%.*]] = call noundef i32 @_Z10int_sourcev()
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i8
    // CHECK-V6COMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = and i8 [[TRUNC]], 3
    // CHECK: [[T1:%.*]] = and i8 [[OLD_VALUE]], -4
    // CHECK: [[T2:%.*]] = or i8 [[T1]], [[T0]]
    // CHECK-V6COMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: store i8 [[T2]], i8* [[FIELD_P]], align 4
    c.onebit = int_source();

    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B:%.*]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK-V6COMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 2
    // CHECK-NOCOMPAT: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 4
    // CHECK: [[T0:%.*]] = shl i8 [[VALUE]], 6
    // CHECK: [[T1:%.*]] = ashr i8 [[T0]], 6
    // CHECK: [[T2:%.*]] = sext i8 [[T1]] to i32
    // CHECK: call void @_Z8int_sinki(i32 noundef [[T2]])
    int_sink(c.onebit);
  }

  // These accesses should have alignment 8 because they're at offset 8
  // in an alignment-16 variable.
  // CHECK-LABEL: @_ZN5test01eEv
  void e() {
    // CHECK: [[C_P:%.*]] = alloca [[C:%.*]], align 16
    __attribute__((aligned(16))) C c;

    // CHECK: [[CALL:%.*]] = call noundef i32 @_Z10int_sourcev()
    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i8
    // CHECK: [[OLD_VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 8
    // CHECK: [[T0:%.*]] = and i8 [[TRUNC]], 3
    // CHECK: [[T1:%.*]] = and i8 [[OLD_VALUE]], -4
    // CHECK: [[T2:%.*]] = or i8 [[T1]], [[T0]]
    // CHECK: store i8 [[T2]], i8* [[FIELD_P]], align 8
    c.onebit = int_source();

    // CHECK: [[T0:%.*]] = bitcast [[C]]* [[C_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 8
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B:%.*]]*
    // CHECK: [[FIELD_P:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[VALUE:%.*]] = load i8, i8* [[FIELD_P]], align 8
    // CHECK: [[T0:%.*]] = shl i8 [[VALUE]], 6
    // CHECK: [[T1:%.*]] = ashr i8 [[T0]], 6
    // CHECK: [[T2:%.*]] = sext i8 [[T1]] to i32
    // CHECK: call void @_Z8int_sinki(i32 noundef [[T2]])
    int_sink(c.onebit);
  }
}

namespace test1 {
  struct Array {
    int elts[4];
  };

  struct A {
    __attribute__((aligned(16))) Array aArray;
  };

  struct B : virtual A {
    void *bPointer; // puts bArray at offset 16
    Array bArray;
  };

  struct C : virtual A { // must be viable as primary base
    // Non-empty, nv-size not a multiple of 16.
    void *cPointer1;
    void *cPointer2;
  };

  // Proof of concept that the non-virtual components of B do not have
  // to be 16-byte-aligned.
  struct D : C, B {};

  // For the following tests, we want to assign into a variable whose
  // alignment is high enough that it will absolutely not be the
  // constraint on the memcpy alignment.
  typedef __attribute__((aligned(64))) Array AlignedArray;

  // CHECK-LABEL: @_ZN5test11aERNS_1AE
  void a(A &a) {
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY:%.*]], align 64
    // CHECK: [[A_P:%.*]] = load [[A:%.*]]*, [[A]]**
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A_P]], i32 0, i32 0
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 16 [[T1]], i64 16, i1 false)
    AlignedArray result = a.aArray;
  }

  // CHECK-LABEL: @_ZN5test11bERNS_1BE
  void b(B &b) {
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[B_P:%.*]] = load [[B:%.*]]*, [[B]]**
    // CHECK: [[VPTR_P:%.*]] = bitcast [[B]]* [[B_P]] to i8**
    // CHECK: [[VPTR:%.*]] = load i8*, i8** [[VPTR_P]], align 8
    // CHECK: [[T0:%.*]] = getelementptr i8, i8* [[VPTR]], i64 -24
    // CHECK: [[OFFSET_P:%.*]] = bitcast i8* [[T0]] to i64*
    // CHECK: [[OFFSET:%.*]] = load i64, i64* [[OFFSET_P]], align 8
    // CHECK: [[T0:%.*]] = bitcast [[B]]* [[B_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[OFFSET]]
    // CHECK: [[A_P:%.*]] = bitcast i8* [[T1]] to [[A]]*
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A_P]], i32 0, i32 0
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 16 [[T1]], i64 16, i1 false)
    AlignedArray result = b.aArray;
  }

  // CHECK-LABEL: @_ZN5test11cERNS_1BE
  void c(B &b) {
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[B_P:%.*]] = load [[B]]*, [[B]]**
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[B]], [[B]]* [[B_P]], i32 0, i32 2
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 8 [[T1]], i64 16, i1 false)
    AlignedArray result = b.bArray;
  }

  // CHECK-LABEL: @_ZN5test11dEPNS_1BE
  void d(B *b) {
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[B_P:%.*]] = load [[B]]*, [[B]]**
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[B]], [[B]]* [[B_P]], i32 0, i32 2
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 8 [[T1]], i64 16, i1 false)
    AlignedArray result = b->bArray;
  }

  // CHECK-LABEL: @_ZN5test11eEv
  void e() {
    // CHECK: [[B_P:%.*]] = alloca [[B]], align 16
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[B]], [[B]]* [[B_P]], i32 0, i32 2
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 16 [[T1]], i64 16, i1 false)
    B b;
    AlignedArray result = b.bArray;
  }

  // CHECK-LABEL: @_ZN5test11fEv
  void f() {
    // TODO: we should devirtualize this derived-to-base conversion.
    // CHECK: [[D_P:%.*]] = alloca [[D:%.*]], align 16
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[VPTR_P:%.*]] = bitcast [[D]]* [[D_P]] to i8**
    // CHECK: [[VPTR:%.*]] = load i8*, i8** [[VPTR_P]], align 16
    // CHECK: [[T0:%.*]] = getelementptr i8, i8* [[VPTR]], i64 -24
    // CHECK: [[OFFSET_P:%.*]] = bitcast i8* [[T0]] to i64*
    // CHECK: [[OFFSET:%.*]] = load i64, i64* [[OFFSET_P]], align 8
    // CHECK: [[T0:%.*]] = bitcast [[D]]* [[D_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 [[OFFSET]]
    // CHECK: [[A_P:%.*]] = bitcast i8* [[T1]] to [[A]]*
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A_P]], i32 0, i32 0
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 16 [[T1]], i64 16, i1 false)
    D d;
    AlignedArray result = d.aArray;
  }

  // CHECK-LABEL: @_ZN5test11gEv
  void g() {
    // CHECK: [[D_P:%.*]] = alloca [[D]], align 16
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[T0:%.*]] = bitcast [[D]]* [[D_P]] to i8*
    // CHECK: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i64 24
    // CHECK: [[B_P:%.*]] = bitcast i8* [[T1]] to [[B:%.*]]*
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[B]], [[B]]* [[B_P]], i32 0, i32 2
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 8 [[T1]], i64 16, i1 false)
    D d;
    AlignedArray result = d.bArray;
  }

  // CHECK-LABEL: @_ZN5test11hEPA_NS_1BE
  void h(B (*b)[]) {
    // CHECK: [[RESULT:%.*]] = alloca [[ARRAY]], align 64
    // CHECK: [[B_P:%.*]] = load [0 x [[B]]]*, [0 x [[B]]]**
    // CHECK: [[ELEMENT_P:%.*]] = getelementptr inbounds [0 x [[B]]], [0 x [[B]]]* [[B_P]], i64 0
    // CHECK: [[ARRAY_P:%.*]] = getelementptr inbounds [[B]], [[B]]* [[ELEMENT_P]], i32 0, i32 2
    // CHECK: [[T0:%.*]] = bitcast [[ARRAY]]* [[RESULT]] to i8*
    // CHECK: [[T1:%.*]] = bitcast [[ARRAY]]* [[ARRAY_P]] to i8*
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 64 [[T0]], i8* align 16 [[T1]], i64 16, i1 false)
    AlignedArray result = (*b)->bArray;
  }
}

// CHECK-LABEL: @_Z22incomplete_array_derefPA_i
// CHECK: load i32, i32* {{%.*}}, align 4
int incomplete_array_deref(int (*p)[]) { return (*p)[2]; }
