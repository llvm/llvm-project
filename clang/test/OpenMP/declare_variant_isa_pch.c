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

//--- variants.h
void avx_variant(void);
#pragma omp declare variant(avx_variant) match(device={isa(avx512f)})
void base_isa(void);

//--- use.c
// expected-no-diagnostics
void caller(void) { base_isa(); }
