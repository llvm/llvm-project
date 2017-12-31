// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp -triple i386-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC
// RUN: %clang_cc1 -fopenmp -triple powerpc64-unknown-unknown -target-abi elfv1-qpx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC-QPX

// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=X86-AVX512
// RUN: %clang_cc1 -fopenmp-simd -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC
// RUN: %clang_cc1 -fopenmp-simd -triple powerpc64-unknown-unknown -target-abi elfv1-qpx -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=PPC-QPX

void h1(float *c, float *a, double b[], int size)
{
// CHECK-LABEL: define void @h1
  int t = 0;
#pragma omp simd safelen(16) linear(t) aligned(c:32) aligned(a,b)
// CHECK:         [[C_PTRINT:%.+]] = ptrtoint
// CHECK-NEXT:    [[C_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[C_PTRINT]], 31
// CHECK-NEXT:    [[C_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[C_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[C_MASKCOND]])
// CHECK:         [[A_PTRINT:%.+]] = ptrtoint

// X86-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// X86-AVX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 31
// X86-AVX512-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 63
// PPC-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// PPC-QPX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15

// CHECK-NEXT:    [[A_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[A_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[A_MASKCOND]])
// CHECK:         [[B_PTRINT:%.+]] = ptrtoint

// X86-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// X86-AVX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31
// X86-AVX512-NEXT: [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 63
// PPC-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// PPC-QPX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31

// CHECK-NEXT:    [[B_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[B_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[B_MASKCOND]])
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
  }
// do not emit parallel_loop_access metadata due to usage of safelen clause.
// CHECK-NOT: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.mem.parallel_loop_access {{![0-9]+}}
#pragma omp simd safelen(16) linear(t) aligned(c:32) aligned(a,b) simdlen(8)
// CHECK:         [[C_PTRINT:%.+]] = ptrtoint
// CHECK-NEXT:    [[C_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[C_PTRINT]], 31
// CHECK-NEXT:    [[C_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[C_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[C_MASKCOND]])
// CHECK:         [[A_PTRINT:%.+]] = ptrtoint

// X86-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// X86-AVX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 31
// X86-AVX512-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 63
// PPC-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// PPC-QPX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15

// CHECK-NEXT:    [[A_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[A_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[A_MASKCOND]])
// CHECK:         [[B_PTRINT:%.+]] = ptrtoint

// X86-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// X86-AVX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31
// X86-AVX512-NEXT: [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 63
// PPC-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// PPC-QPX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31

// CHECK-NEXT:    [[B_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[B_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[B_MASKCOND]])
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
  }
// do not emit parallel_loop_access metadata due to usage of safelen clause.
// CHECK-NOT: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.mem.parallel_loop_access {{![0-9]+}}
#pragma omp simd linear(t) aligned(c:32) aligned(a,b) simdlen(8)
// CHECK:         [[C_PTRINT:%.+]] = ptrtoint
// CHECK-NEXT:    [[C_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[C_PTRINT]], 31
// CHECK-NEXT:    [[C_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[C_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[C_MASKCOND]])
// CHECK:         [[A_PTRINT:%.+]] = ptrtoint

// X86-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// X86-AVX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 31
// X86-AVX512-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 63
// PPC-NEXT:     [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15
// PPC-QPX-NEXT: [[A_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[A_PTRINT]], 15

// CHECK-NEXT:    [[A_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[A_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[A_MASKCOND]])
// CHECK:         [[B_PTRINT:%.+]] = ptrtoint

// X86-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// X86-AVX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31
// X86-AVX512-NEXT: [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 63
// PPC-NEXT:      [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 15
// PPC-QPX-NEXT:  [[B_MASKEDPTR:%.+]] = and i{{[0-9]+}} [[B_PTRINT]], 31

// CHECK-NEXT:    [[B_MASKCOND:%.+]] = icmp eq i{{[0-9]+}} [[B_MASKEDPTR]], 0
// CHECK-NEXT:    call void @llvm.assume(i1 [[B_MASKCOND]])
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
// CHECK: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.mem.parallel_loop_access {{![0-9]+}}
  }
}

void h2(float *c, float *a, float *b, int size)
{
// CHECK-LABEL: define void @h2
  int t = 0;
#pragma omp simd linear(t)
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] * a[i] + b[i] * b[t];
    ++t;
// CHECK: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.mem.parallel_loop_access [[LOOP_H2_HEADER:![0-9]+]]
  }
}

void h3(float *c, float *a, float *b, int size)
{
// CHECK-LABEL: define void @h3
#pragma omp simd
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      c[j*i] = a[i] * b[j];
    }
  }
// do not emit parallel_loop_access for nested loop.
// CHECK-NOT: store float {{.+}}, float* {{.+}}, align {{.+}}, !llvm.mem.parallel_loop_access {{![0-9]+}}
}

// Metadata for h1:
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], [[LOOP_WIDTH_16:![0-9]+]], [[LOOP_VEC_ENABLE:![0-9]+]]}
// CHECK: [[LOOP_WIDTH_16]] = !{!"llvm.loop.vectorize.width", i32 16}
// CHECK: [[LOOP_VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], [[LOOP_WIDTH_8:![0-9]+]], [[LOOP_VEC_ENABLE]]}
// CHECK: [[LOOP_WIDTH_8]] = !{!"llvm.loop.vectorize.width", i32 8}
// CHECK: [[LOOP_H1_HEADER:![0-9]+]] = distinct !{[[LOOP_H1_HEADER]], [[LOOP_WIDTH_8]], [[LOOP_VEC_ENABLE]]}
//
// Metadata for h2:
// CHECK: [[LOOP_H2_HEADER]] = distinct !{[[LOOP_H2_HEADER]], [[LOOP_VEC_ENABLE]]}
//
// Metadata for h3:
// CHECK: [[LOOP_H3_HEADER:![0-9]+]] = distinct !{[[LOOP_H3_HEADER]], [[LOOP_VEC_ENABLE]]}
//
