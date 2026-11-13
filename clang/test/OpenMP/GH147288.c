// RUN: %clang_cc1 -verify -fopenmp -debug-info-kind=standalone -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,PPC64
// RUN: %clang_cc1 -verify -fopenmp -debug-info-kind=standalone -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,PPC64
// RUN: %clang_cc1 -verify -fopenmp -debug-info-kind=standalone -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,X86_64
// RUN: %clang_cc1 -verify -fopenmp -debug-info-kind=standalone -triple s390x-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,S390X
// expected-no-diagnostics

// With debug info, the `.omp_outlined` wrapper (uintptr parameters) calls
// `.omp_outlined_debug__` (original parameter types) and must pass each
// argument as the target ABI lowers it: two i32 for `_Complex int` on
// PowerPC64, indirectly on SystemZ, coerced to i64 for `_BitInt(33)` on x86-64.

void sink_mixed(int, int _Complex, float);
void sink_complex_fd(float _Complex, double _Complex);
void sink_bitint(_BitInt(33));

// CHECK-LABEL: define internal void @apply_round_stamp.omp_outlined_debug__(
// PPC64-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, i32 noundef %{{[^,)]+}})
// X86_64-SAME:   ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
// S390X-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
//
// CHECK-LABEL: define internal void @apply_round_stamp.omp_outlined(
// CHECK-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
// PPC64:         call void @apply_round_stamp.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, i32 noundef %{{[^,)]+}})
// X86_64:        call void @apply_round_stamp.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
// S390X:         call void @apply_round_stamp.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, ptr {{[^%]*}}%byval-temp)
void apply_round_stamp(void) {
  int _Complex strength = 0;
#pragma omp parallel default(firstprivate)
  (void)strength;
}

// Arguments after the complex one must still land in their own parameters.
// CHECK-LABEL: define internal void @mixed.omp_outlined_debug__(
// PPC64-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i32 noundef signext %{{[^,]+}}, i32 noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, float noundef %{{[^,)]+}})
// X86_64-SAME:   ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, i64 noundef %{{[^,]+}}, float noundef %{{[^,)]+}})
//
// CHECK-LABEL: define internal void @mixed.omp_outlined(
// CHECK-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i64 noundef %{{[^,]+}}, i64 noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
// PPC64:         call void @mixed.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i32 noundef signext %{{[^,]+}}, i32 noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, float noundef %{{[^,)]+}})
// X86_64:        call void @mixed.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, i64 noundef %{{[^,]+}}, float noundef %{{[^,)]+}})
void mixed(void) {
  int a = 1;
  int _Complex c = 0;
  float x = 2;
#pragma omp parallel default(firstprivate)
  sink_mixed(a, c, x);
}

// `_Complex float` is captured by copy; `_Complex double` does not fit in a
// uintptr_t and is captured by reference.
// CHECK-LABEL: define internal void @complex_fd.omp_outlined_debug__(
// PPC64-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, float noundef %{{[^,]+}}, float noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
// X86_64-SAME:   ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, <2 x float> noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
//
// CHECK-LABEL: define internal void @complex_fd.omp_outlined(
// CHECK-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i64 noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
// PPC64:         call void @complex_fd.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, float noundef %{{[^,]+}}, float noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
// X86_64:        call void @complex_fd.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, <2 x float> noundef %{{[^,]+}}, ptr {{[^%]*}}%{{[^,)]+}})
void complex_fd(void) {
  float _Complex f = 0;
  double _Complex d = 0;
#pragma omp parallel firstprivate(f, d)
  sink_complex_fd(f, d);
}

// CHECK-LABEL: define internal void @bitint.omp_outlined_debug__(
// PPC64-SAME:    ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i33 noundef signext %{{[^,)]+}})
// X86_64-SAME:   ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
//
// CHECK-LABEL: define internal void @bitint.omp_outlined(
// PPC64:         call void @bitint.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i33 noundef signext %{{[^,)]+}})
// X86_64:        call void @bitint.omp_outlined_debug__(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i64 noundef %{{[^,)]+}})
void bitint(void) {
  _BitInt(33) b = 5;
#pragma omp parallel default(firstprivate)
  sink_bitint(b);
}
