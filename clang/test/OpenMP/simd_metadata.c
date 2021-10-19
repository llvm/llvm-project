// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC

// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp-simd -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC

void h1(float *c, float *a, double b[], int size)
{
// CHECK-LABEL: define{{.*}} void @h1
  int t = 0;
#pragma omp simd safelen(16) linear(t) aligned(c:32) aligned(a,b)
  // CHECK:         call void @llvm.assume(i1 true) [ "align"(float* [[PTR4:%.*]], {{i64|i32}} 32) ]
  // CHECK-NEXT:    load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // CHECK-NEXT:     load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
  }
// do not emit llvm.access.group metadata due to usage of safelen clause.
// CHECK-NOT: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.access.group {{![0-9]+}}
#pragma omp simd safelen(16) linear(t) aligned(c:32) aligned(a,b) simdlen(8)
  // CHECK:         call void @llvm.assume(i1 true) [ "align"(float* [[PTR4:%.*]], {{i64|i32}} 32) ]
  // CHECK-NEXT:    load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // CHECK-NEXT:     load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
  }
// do not emit llvm.access.group metadata due to usage of safelen clause.
// CHECK-NOT: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.access.group {{![0-9]+}}
#pragma omp simd linear(t) aligned(c:32) aligned(a,b) simdlen(8)
  // CHECK:         call void @llvm.assume(i1 true) [ "align"(float* [[PTR4:%.*]], {{i64|i32}} 32) ]
  // CHECK-NEXT:    load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(float* [[PTR5:%.*]], {{i64|i32}} 16) ]
  // CHECK-NEXT:     load

  // X86-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  // X86-AVX-NEXT:   call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 32) ]
  // X86-AVX512-NEXT:call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 64) ]
  // PPC-NEXT:       call void @llvm.assume(i1 true) [ "align"(double* [[PTR6:%.*]], {{i64|i32}} 16) ]
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
// CHECK: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.access.group ![[ACCESS_GROUP_7:[0-9]+]]
  }
}

void h2(float *c, float *a, float *b, int size)
{
// CHECK-LABEL: define{{.*}} void @h2
  int t = 0;
#pragma omp simd linear(t)
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
// CHECK: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.access.group ![[ACCESS_GROUP_10:[0-9]+]]
  }
// CHECK: br label %{{.+}}, !llvm.loop [[LOOP_H2_HEADER:![0-9]+]]
}

void h3(float *c, float *a, float *b, int size)
{
// CHECK-LABEL: define{{.*}} void @h3
#pragma omp simd
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      c[j*i] = a[i] * b[j];
    }
// CHECK: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.access.group ![[ACCESS_GROUP_13:[0-9]+]]
  }
  // CHECK: br label %{{.+}}, !llvm.loop [[LOOP_H3_HEADER_INNER:![0-9]+]]
  // CHECK: br label %{{.+}}, !llvm.loop [[LOOP_H3_HEADER:![0-9]+]]
}

// Metadata for h1:
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], [[LOOP_WIDTH_16:![0-9]+]], [[LOOP_VEC_ENABLE:![0-9]+]]}
// CHECK: [[LOOP_WIDTH_16]] = !{!"llvm.loop.vectorize.width", i32 16}
// CHECK: [[LOOP_VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], [[LOOP_WIDTH_8:![0-9]+]], [[LOOP_VEC_ENABLE]]}
// CHECK: [[LOOP_WIDTH_8]] = !{!"llvm.loop.vectorize.width", i32 8}
// CHECK: ![[ACCESS_GROUP_7]] = distinct !{}
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], ![[PARALLEL_ACCESSES_9:[0-9]+]], [[LOOP_WIDTH_8]], [[LOOP_VEC_ENABLE]]}
// CHECK: ![[PARALLEL_ACCESSES_9]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_7]]}
//
// Metadata for h2:
// CHECK: ![[ACCESS_GROUP_10]] = distinct !{}
// CHECK: [[LOOP_H2_HEADER]] = distinct !{[[LOOP_H2_HEADER]], ![[PARALLEL_ACCESSES_12:[0-9]+]], [[LOOP_VEC_ENABLE]]}
// CHECK: ![[PARALLEL_ACCESSES_12]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_10]]}
//
// Metadata for h3:
// CHECK: ![[ACCESS_GROUP_13]] = distinct !{}
// CHECK: [[LOOP_H3_HEADER]] = distinct !{[[LOOP_H3_HEADER]], ![[PARALLEL_ACCESSES_15:[0-9]+]], [[LOOP_VEC_ENABLE]]}
// CHECK: ![[PARALLEL_ACCESSES_15]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_13]]}
//
