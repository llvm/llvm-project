// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -target-feature +avx \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -verify -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -target-feature +avx \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -E -fopenmp -fopenmp-version=52 %s -o %t.i
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -target-feature +avx \
// RUN:   -emit-llvm %t.i -o - | FileCheck %s
// expected-no-diagnostics

#ifdef __cplusplus
extern "C" {
#endif

#pragma omp begin declare target
void cpu_variant(void);
void arch_variant(void);
void scored_variant(void);
void parallel_variant(void);

#pragma omp declare variant(cpu_variant) match(device = {kind(cpu)})
#pragma omp declare variant(scored_variant) \
    match(implementation = {vendor(score(3) : llvm)})
void target_base(void);

#pragma omp declare variant(cpu_variant) match(device = {kind(cpu)})
#pragma omp declare variant(parallel_variant) match(construct = {parallel})
void depth_base(void);

#pragma omp declare variant(parallel_variant) match(construct = {parallel})
void construct_base(void);

#pragma omp declare variant(cpu_variant) match(device = {kind(cpu)})
#pragma omp declare variant(scored_variant) \
    match(implementation = {vendor(score(3) : llvm)})
void task_depth_base(void);

#pragma omp declare variant(arch_variant) match(device = {arch(x86_64)})
#pragma omp declare variant(scored_variant) \
    match(implementation = {vendor(score(3) : llvm)})
void source_depth_base(void);

int arch_value(void);
int scored_value(void);
#pragma omp declare variant(arch_value) match(device = {arch(x86_64)})
#pragma omp declare variant(scored_value) \
    match(implementation = {vendor(score(3) : llvm)})
int source_depth_value(void);

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

// Macro spelling must not hide different expanded conditions.
#define COND 1
#pragma omp declare variant(condition_high_variant)                       \
    match(implementation = {vendor(score(100) : llvm)},                    \
          user = {condition(COND)})
#undef COND
#define COND 2
#pragma omp declare variant(condition_low_variant)                        \
    match(implementation = {vendor(score(1) : llvm)}, device = {kind(cpu)}, \
          user = {condition(COND)})
void redefined_condition_base(void);
#undef COND

// Different macro names and outer parentheses preserve condition identity.
#define FIRST_COND 1
#define SECOND_COND (1)
#pragma omp declare variant(condition_high_variant)                       \
    match(implementation = {vendor(score(100) : llvm)},                    \
          user = {condition(FIRST_COND)})
#pragma omp declare variant(condition_low_variant)                        \
    match(implementation = {vendor(score(1) : llvm)}, device = {kind(cpu)}, \
          user = {condition(SECOND_COND)})
void equivalent_condition_base(void);
#undef FIRST_COND
#undef SECOND_COND

void subset_variant(void);
void superset_variant(void);
#pragma omp declare variant(subset_variant) \
    match(implementation = {vendor(score(100) : llvm)})
#pragma omp declare variant(superset_variant) \
    match(implementation = {vendor(score(1) : llvm)}, user = {condition(1)})
void subset_base(void);

#pragma omp declare variant(subset_variant) \
    match(implementation = {vendor(score(100) : llvm)})
#pragma omp declare variant(superset_variant) \
    match(implementation = {vendor(score(1) : llvm)}, device = {kind(any)})
void any_base(void);

#pragma omp declare variant(cpu_variant) match(device = {kind(cpu)})
#pragma omp declare variant(scored_variant) \
    match(implementation = {vendor(score(5) : llvm)})
void sections_base(void);

void ordered_high_variant(void);
void ordered_low_variant(void);
#pragma omp declare variant(ordered_high_variant)                            \
    match(construct = {parallel, for},                                       \
          implementation = {vendor(score(100) : llvm)})
#pragma omp declare variant(ordered_low_variant)                             \
    match(construct = {for, parallel, simd},                                 \
          implementation = {vendor(score(1) : llvm)})
void ordered_base(void);

void isa_high_variant(void);
void isa_low_variant(void);
#pragma omp declare variant(isa_high_variant)                                \
    match(device = {isa("sse2")},                                           \
          implementation = {vendor(score(100) : llvm)})
#pragma omp declare variant(isa_low_variant)                                 \
    match(device = {isa("avx")}, implementation = {vendor(score(1) : llvm)}, \
          user = {condition(1)})
void isa_base(void);
#pragma omp end declare target

// With only TARGET in the context, CPU scores 3 and the vendor variant's
// total score is 4.
void target_only(void) {
#pragma omp target
  { target_base(); }
}
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}target_only
// CHECK: call void @scored_variant()
// CHECK: ret void

// The outer PARALLEL must not increase the device score inside TARGET.
void parallel_target(void) {
#pragma omp parallel
  {
#pragma omp target
    { target_base(); }
  }
}
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}parallel_target
// CHECK: call void @scored_variant()
// CHECK: ret void

// In PARALLEL, CPU scores 3 and construct={parallel} scores 2.
void parallel_depth(void) {
#pragma omp parallel
  { depth_base(); }
}
// CHECK-LABEL: define internal void @parallel_depth.omp_outlined
// CHECK: call void @cpu_variant()
// CHECK: ret void

// TARGET is retained, as are constructs nested inside it: CPU scores 5.
void target_parallel(void) {
#pragma omp target
  {
#pragma omp parallel
    { target_base(); }
  }
}
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}target_parallel
// CHECK: define internal void @{{.*}}omp_outlined
// CHECK: call void @cpu_variant()
// CHECK: ret void

// Leaving TARGET restores the enclosing PARALLEL context.
void restore_context(void) {
#pragma omp parallel
  {
#pragma omp target
    { target_base(); }
    depth_base();
    construct_base();
  }
}
// CHECK-LABEL: define internal void @restore_context.omp_outlined
// CHECK: call void @cpu_variant()
// CHECK: call void @parallel_variant()
// CHECK: ret void
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}restore_context
// CHECK: call void @scored_variant()
// CHECK: ret void

// TASK has no construct-selector property, but contributes to the device
// weight. In PARALLEL > TASK, CPU scores 5 and beats the vendor variant's
// total score of 4.
void task_depth(void) {
#pragma omp parallel
  {
#pragma omp task
    { task_depth_base(); }
  }
}
// CHECK-LABEL: define internal {{.*}}i32 @.omp_task_entry.
// CHECK: call void @cpu_variant()

// Executable loop transformations contribute to the construct depth. At
// depth one, ARCH scores 5 and beats the vendor variant's score of 4.
void tile_depth(void) {
#pragma omp tile sizes(2)
  for (int i = 0; i < 4; ++i)
    source_depth_base();
}
// CHECK-LABEL: define{{.*}} void @tile_depth
// CHECK: call void @arch_variant()
// CHECK: ret void

void unroll_depth(void) {
#pragma omp unroll partial(2)
  for (int i = 0; i < 4; ++i)
    source_depth_base();
}
// CHECK-LABEL: define{{.*}} void @unroll_depth
// CHECK: call void @arch_variant()
// CHECK: ret void

// ATOMIC is an executable construct and contributes one context position.
void atomic_depth(int *x) {
#pragma omp atomic update
  *x += source_depth_value();
}
// CHECK-LABEL: define{{.*}} void @atomic_depth
// CHECK: call i32 @arch_value()
// CHECK: ret void

// ASSUME is informational and does not add a position. Inside PARALLEL, CPU
// scores 3 and loses to the vendor variant's score of 4.
void assume_context(void) {
#pragma omp parallel
  {
#pragma omp assume no_openmp_routines
    { task_depth_base(); }
  }
}
// CHECK-LABEL: define internal void @assume_context.omp_outlined
// CHECK: call void @scored_variant()
// CHECK: ret void

// Distinct folded user conditions do not create a subset relationship.
void distinct_conditions(void) { distinct_condition_base(); }
// CHECK-LABEL: define{{.*}} void @distinct_conditions
// CHECK: call void @condition_high_variant()
// CHECK: ret void

// Identical conditions allow the first selector to be a strict subset.
void identical_conditions(void) { identical_condition_base(); }
// CHECK-LABEL: define{{.*}} void @identical_conditions
// CHECK: call void @condition_low_variant()
// CHECK: ret void

void redefined_conditions(void) { redefined_condition_base(); }
// CHECK-LABEL: define{{.*}} void @redefined_conditions
// CHECK-NOT: call void @condition_low_variant()
// CHECK: call void @condition_high_variant()
// CHECK-NOT: call void @condition_low_variant()
// CHECK: ret void

void equivalent_conditions(void) { equivalent_condition_base(); }
// CHECK-LABEL: define{{.*}} void @equivalent_conditions
// CHECK-NOT: call void @condition_high_variant()
// CHECK: call void @condition_low_variant()
// CHECK-NOT: call void @condition_high_variant()
// CHECK: ret void

// A strict subset has score zero before candidates are ranked, even when its
// explicit score would otherwise be higher.
void strict_subset(void) { subset_base(); }
// CHECK-LABEL: define{{.*}} void @strict_subset
// CHECK: call void @superset_variant()
// CHECK: ret void

// kind(any) does not make the lower-scored selector a strict superset.
void kind_any(void) { any_base(); }
// CHECK-LABEL: define{{.*}} void @kind_any
// CHECK: call void @subset_variant()
// CHECK: ret void

// SECTION is a separator, so CPU scores 5 and the vendor scores 6 in both
// spellings of the first section of PARALLEL SECTIONS.
void explicit_section(void) {
#pragma omp parallel sections
  {
#pragma omp section
    { sections_base(); }
  }
}
// CHECK-LABEL: define internal void @explicit_section.omp_outlined
// CHECK: call void @scored_variant()
// CHECK: ret void

void implicit_section(void) {
#pragma omp parallel sections
  { sections_base(); }
}
// CHECK-LABEL: define internal void @implicit_section.omp_outlined
// CHECK: call void @scored_variant()
// CHECK: ret void

// Different construct orders do not create a subset relationship.
void different_construct_order(void) {
#pragma omp parallel for
  for (int i = 0; i < 2; ++i) {
#pragma omp parallel for simd
    for (int j = 0; j < 2; ++j)
      ordered_base();
  }
}
// CHECK-LABEL: define internal void @different_construct_order.omp_outlined
// CHECK: define internal void @{{.*}}omp_outlined
// CHECK: call void @ordered_high_variant()
// CHECK: ret void

// Different active ISA properties do not create a subset relationship.
void different_isa(void) { isa_base(); }
// CHECK-LABEL: define{{.*}} void @different_isa
// CHECK: call void @isa_high_variant()
// CHECK: ret void

#ifdef __cplusplus
}
#endif
