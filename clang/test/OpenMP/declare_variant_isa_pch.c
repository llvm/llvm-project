// RUN: split-file %s %t
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux -target-feature +avx512f \
// RUN:   -x c-header %t/variants.h -emit-pch -o %t/variants.pch
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux -target-feature +avx512f \
// RUN:   -include-pch %t/variants.pch %t/use.c -emit-llvm -o - -verify | \
// RUN:   FileCheck %s --check-prefix=AVX512
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux \
// RUN:   -x c-header %t/variants.h -emit-pch -o %t/generic.pch
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux \
// RUN:   -include-pch %t/generic.pch %t/use.c -emit-llvm -o - -verify | \
// RUN:   FileCheck %s --check-prefix=GENERIC

// Resolve the call after loading the selector, not while building the PCH.
// AVX512-LABEL: define{{.*}} void @caller(
// AVX512: call void @avx_variant()
// AVX512-NOT: call void @base_isa()
// AVX512: ret void
// GENERIC-LABEL: define{{.*}} void @caller(
// GENERIC: call void @base_isa()
// GENERIC-NOT: call void @avx_variant()
// GENERIC: ret void

// Different ISA names are not subsets of each other, so the higher vendor score
// wins with match_any whether or not the target has AVX-512.
// AVX512-LABEL: define{{.*}} void @distinct_isa_caller(
// AVX512: call void @avx512_any()
// AVX512-NOT: call void @sse2_any()
// AVX512: ret void
// GENERIC-LABEL: define{{.*}} void @distinct_isa_caller(
// GENERIC: call void @avx512_any()
// GENERIC-NOT: call void @sse2_any()
// GENERIC: ret void

//--- variants.h
void avx_variant(void);
#pragma omp declare variant(avx_variant) match(device={isa(avx512f)})
void base_isa(void);

void avx512_any(void);
void sse2_any(void);
#pragma omp declare variant(avx512_any) \
    match(device={isa(avx512f)}, \
          implementation={vendor(score(100): llvm), extension(match_any)})
#pragma omp declare variant(sse2_any) \
    match(device={isa(sse2)}, \
          implementation={vendor(score(1): llvm), extension(match_any)}, \
          user={condition(1)})
void distinct_isa_base(void);

//--- use.c
// expected-no-diagnostics
void caller(void) { base_isa(); }
void distinct_isa_caller(void) { distinct_isa_base(); }
