// RUN: split-file %s %t
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -x c-header %t/conditions.h \
// RUN:   -emit-pch -o %t/conditions-c.pch
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -x c %t/use.c \
// RUN:   -include-pch %t/conditions-c.pch -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -x c++-header %t/conditions.h \
// RUN:   -emit-pch -o %t/conditions-cxx.pch
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -x c++ %t/use.c \
// RUN:   -include-pch %t/conditions-cxx.pch -emit-llvm -o - | FileCheck %s

// Verify that serialization preserves the source identity of folded user
// conditions. Distinct conditions do not form a subset relationship, while
// identical conditions do.

// CHECK-LABEL: define{{.*}} void @test_conditions()
// CHECK: call void @condition_high_variant()
// CHECK-NEXT: call void @condition_low_variant()
// CHECK: ret void

//--- conditions.h
#ifdef __cplusplus
extern "C" {
#endif

void condition_high_variant(void);
void condition_low_variant(void);

#pragma omp declare variant(condition_high_variant)                       \
    match(implementation = {vendor(score(100) : llvm)}, user = {condition(1)})
#pragma omp declare variant(condition_low_variant)                         \
    match(implementation = {vendor(score(1) : llvm)}, device = {kind(cpu)}, \
          user = {condition(2)})
void distinct_condition_base(void);

#pragma omp declare variant(condition_high_variant)                       \
    match(implementation = {vendor(score(100) : llvm)}, user = {condition(1)})
#pragma omp declare variant(condition_low_variant)                         \
    match(implementation = {vendor(score(1) : llvm)}, device = {kind(cpu)}, \
          user = {condition(1)})
void identical_condition_base(void);

#ifdef __cplusplus
}
#endif

//--- use.c
#ifdef __cplusplus
extern "C"
#endif
void test_conditions(void) {
  distinct_condition_base();
  identical_condition_base();
}
