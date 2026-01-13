// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR --check-prefix=CIR-NO-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM --check-prefix=LLVM-NO-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG --check-prefix=OGCG-NO-STRICT

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=0 -emit-cir %s -o %t-strict-0.cir
// RUN: FileCheck --input-file=%t-strict-0.cir %s --check-prefix=CIR --check-prefix=CIR-STRICT-0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=0 -emit-llvm -disable-llvm-passes %s -o %t-cir-strict-0.ll
// RUN: FileCheck --input-file=%t-cir-strict-0.ll %s --check-prefix=LLVM --check-prefix=LLVM-STRICT-0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fstrict-flex-arrays=0 -emit-llvm -disable-llvm-passes %s -o %t-strict-0.ll
// RUN: FileCheck --input-file=%t-strict-0.ll %s --check-prefix=OGCG --check-prefix=OGCG-STRICT-0

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=1 -emit-cir %s -o %t-strict-1.cir
// RUN: FileCheck --input-file=%t-strict-1.cir %s --check-prefix=CIR --check-prefix=CIR-STRICT-1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=1 -emit-llvm -disable-llvm-passes %s -o %t-cir-strict-1.ll
// RUN: FileCheck --input-file=%t-cir-strict-1.ll %s --check-prefix=LLVM --check-prefix=LLVM-STRICT-1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fstrict-flex-arrays=1 -emit-llvm -disable-llvm-passes %s -o %t-strict-1.ll
// RUN: FileCheck --input-file=%t-strict-1.ll %s --check-prefix=OGCG --check-prefix=OGCG-STRICT-1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=2 -emit-cir %s -o %t-strict-2.cir
// RUN: FileCheck --input-file=%t-strict-2.cir %s --check-prefix=CIR --check-prefix=CIR-STRICT-2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=2 -emit-llvm -disable-llvm-passes %s -o %t-cir-strict-2.ll
// RUN: FileCheck --input-file=%t-cir-strict-2.ll %s --check-prefix=LLVM --check-prefix=LLVM-STRICT-2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fstrict-flex-arrays=2 -emit-llvm -disable-llvm-passes %s -o %t-strict-2.ll
// RUN: FileCheck --input-file=%t-strict-2.ll %s --check-prefix=OGCG --check-prefix=OGCG-STRICT-2

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=3 -emit-cir %s -o %t-strict-3.cir
// RUN: FileCheck --input-file=%t-strict-3.cir %s --check-prefix=CIR --check-prefix=CIR-STRICT-3
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -fstrict-flex-arrays=3 -emit-llvm -disable-llvm-passes %s -o %t-cir-strict-3.ll
// RUN: FileCheck --input-file=%t-cir-strict-3.ll %s --check-prefix=LLVM --check-prefix=LLVM-STRICT-3
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fstrict-flex-arrays=3 -emit-llvm -disable-llvm-passes %s -o %t-strict-3.ll
// RUN: FileCheck --input-file=%t-strict-3.ll %s --check-prefix=OGCG --check-prefix=OGCG-STRICT-3

#define OBJECT_SIZE_BUILTIN __builtin_object_size

typedef struct {
  float f;
  double c[];
} foo_t;

typedef struct {
  float f;
  double c[0];
} foo0_t;

typedef struct {
  float f;
  double c[1];
} foo1_t;

typedef struct {
  float f;
  double c[2];
} foo2_t;

// CIR-LABEL: @bar
// LLVM-LABEL: @bar(
// OGCG-LABEL: @bar(
unsigned bar(foo_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-3: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-3: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-3: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @bar0
// LLVM-LABEL: @bar0(
// OGCG-LABEL: @bar0(
unsigned bar0(foo0_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-3: cir.const #cir.int<0>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-3: store i32 0
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-3: ret i32 0
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @bar1
// LLVM-LABEL: @bar1(
// OGCG-LABEL: @bar1(
unsigned bar1(foo1_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.const #cir.int<8>
  // CIR-STRICT-3: cir.const #cir.int<8>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-2: store i32 8
  // LLVM-STRICT-3: store i32 8
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-2: ret i32 8
  // OGCG-STRICT-3: ret i32 8
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @bar2
// LLVM-LABEL: @bar2(
// OGCG-LABEL: @bar2(
unsigned bar2(foo2_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.const #cir.int<16>
  // CIR-STRICT-2: cir.const #cir.int<16>
  // CIR-STRICT-3: cir.const #cir.int<16>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // LLVM-STRICT-1: store i32 16
  // LLVM-STRICT-2: store i32 16
  // LLVM-STRICT-3: store i32 16
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  // OGCG-STRICT-1: ret i32 16
  // OGCG-STRICT-2: ret i32 16
  // OGCG-STRICT-3: ret i32 16
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

#define DYNAMIC_OBJECT_SIZE_BUILTIN __builtin_dynamic_object_size

// CIR-LABEL: @dyn_bar
// LLVM-LABEL: @dyn_bar(
// OGCG-LABEL: @dyn_bar(
unsigned dyn_bar(foo_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-3: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-3: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-3: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @dyn_bar0
// LLVM-LABEL: @dyn_bar0(
// OGCG-LABEL: @dyn_bar0(
unsigned dyn_bar0(foo0_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-3: cir.const #cir.int<0>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-3: store i32 0
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-2: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-3: ret i32 0
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @dyn_bar1
// LLVM-LABEL: @dyn_bar1(
// OGCG-LABEL: @dyn_bar1(
unsigned dyn_bar1(foo1_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-2: cir.const #cir.int<8>
  // CIR-STRICT-3: cir.const #cir.int<8>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-2: store i32 8
  // LLVM-STRICT-3: store i32 8
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-1: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-2: ret i32 8
  // OGCG-STRICT-3: ret i32 8
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @dyn_bar2
// LLVM-LABEL: @dyn_bar2(
// OGCG-LABEL: @dyn_bar2(
unsigned dyn_bar2(foo2_t *f) {
  // CIR-NO-STRICT: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-0: cir.objsize max nullunknown dynamic {{.*}} : !cir.ptr<!void> -> !u64i
  // CIR-STRICT-1: cir.const #cir.int<16>
  // CIR-STRICT-2: cir.const #cir.int<16>
  // CIR-STRICT-3: cir.const #cir.int<16>
  // LLVM-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // LLVM-STRICT-1: store i32 16
  // LLVM-STRICT-2: store i32 16
  // LLVM-STRICT-3: store i32 16
  // OGCG-NO-STRICT: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-0: llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true)
  // OGCG-STRICT-1: ret i32 16
  // OGCG-STRICT-2: ret i32 16
  // OGCG-STRICT-3: ret i32 16
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// Also checks for non-trailing flex-array like members

typedef struct {
  double c[0];
  float f;
} foofoo0_t;

typedef struct {
  double c[1];
  float f;
} foofoo1_t;

typedef struct {
  double c[2];
  float f;
} foofoo2_t;

// CIR-LABEL: @babar0
// LLVM-LABEL: @babar0(
// OGCG-LABEL: @babar0(
unsigned babar0(foofoo0_t *f) {
  // CIR-NO-STRICT: cir.const #cir.int<0>
  // CIR-STRICT-0: cir.const #cir.int<0>
  // CIR-STRICT-1: cir.const #cir.int<0>
  // CIR-STRICT-2: cir.const #cir.int<0>
  // CIR-STRICT-3: cir.const #cir.int<0>
  // LLVM-NO-STRICT: store i32 0
  // LLVM-STRICT-0: store i32 0
  // LLVM-STRICT-1: store i32 0
  // LLVM-STRICT-2: store i32 0
  // LLVM-STRICT-3: store i32 0
  // OGCG-NO-STRICT: ret i32 0
  // OGCG-STRICT-0: ret i32 0
  // OGCG-STRICT-1: ret i32 0
  // OGCG-STRICT-2: ret i32 0
  // OGCG-STRICT-3: ret i32 0
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @babar1
// LLVM-LABEL: @babar1(
// OGCG-LABEL: @babar1(
unsigned babar1(foofoo1_t *f) {
  // CIR-NO-STRICT: cir.const #cir.int<8>
  // CIR-STRICT-0: cir.const #cir.int<8>
  // CIR-STRICT-1: cir.const #cir.int<8>
  // CIR-STRICT-2: cir.const #cir.int<8>
  // CIR-STRICT-3: cir.const #cir.int<8>
  // LLVM-NO-STRICT: store i32 8
  // LLVM-STRICT-0: store i32 8
  // LLVM-STRICT-1: store i32 8
  // LLVM-STRICT-2: store i32 8
  // LLVM-STRICT-3: store i32 8
  // OGCG-NO-STRICT: ret i32 8
  // OGCG-STRICT-0: ret i32 8
  // OGCG-STRICT-1: ret i32 8
  // OGCG-STRICT-2: ret i32 8
  // OGCG-STRICT-3: ret i32 8
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CIR-LABEL: @babar2
// LLVM-LABEL: @babar2(
// OGCG-LABEL: @babar2(
unsigned babar2(foofoo2_t *f) {
  // CIR-NO-STRICT: cir.const #cir.int<16>
  // CIR-STRICT-0: cir.const #cir.int<16>
  // CIR-STRICT-1: cir.const #cir.int<16>
  // CIR-STRICT-2: cir.const #cir.int<16>
  // CIR-STRICT-3: cir.const #cir.int<16>
  // LLVM-NO-STRICT: store i32 16
  // LLVM-STRICT-0: store i32 16
  // LLVM-STRICT-1: store i32 16
  // LLVM-STRICT-2: store i32 16
  // LLVM-STRICT-3: store i32 16
  // OGCG-NO-STRICT: ret i32 16
  // OGCG-STRICT-0: ret i32 16
  // OGCG-STRICT-1: ret i32 16
  // OGCG-STRICT-2: ret i32 16
  // OGCG-STRICT-3: ret i32 16
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}
