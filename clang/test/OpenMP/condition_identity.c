// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -emit-llvm %s -o - | FileCheck %s

// Distinct expressions keep lexical order at 5/5 ties. Identical expressions
// permit the KIND(CPU) selector to win the subset tie-break.

// CHECK-LABEL: define {{.*}} @distinct_constants(
// CHECK-NOT: @__kmpc_taskwait
// CHECK: call {{.*}} @__kmpc_taskyield
// CHECK-NOT: @__kmpc_taskwait
// CHECK: ret void
void distinct_constants(void) {
#pragma omp metadirective \
    when(user={condition(1)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(2)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @identical_constants(
// CHECK-NOT: @__kmpc_taskyield
// CHECK: call {{.*}} @__kmpc_taskwait
// CHECK-NOT: @__kmpc_taskyield
// CHECK: ret void
void identical_constants(void) {
#pragma omp metadirective \
    when(user={condition(1)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(1)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @distinct_builtins(
// CHECK-NOT: @__kmpc_taskwait
// CHECK: call {{.*}} @__kmpc_taskyield
// CHECK-NOT: @__kmpc_taskwait
// CHECK: ret void
void distinct_builtins(void) {
#pragma omp metadirective \
    when(user={condition(__builtin_LINE() > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(__builtin_COLUMN() > 0)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @identical_builtins(
// CHECK-NOT: @__kmpc_taskyield
// CHECK: call {{.*}} @__kmpc_taskwait
// CHECK-NOT: @__kmpc_taskyield
// CHECK: ret void
void identical_builtins(void) {
#pragma omp metadirective \
    when(user={condition(__builtin_LINE() > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(__builtin_LINE() > 0)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @distinct_compound_types(
// CHECK-NOT: @__kmpc_taskwait
// CHECK: call {{.*}} @__kmpc_taskyield
// CHECK-NOT: @__kmpc_taskwait
// CHECK: ret void
void distinct_compound_types(void) {
#pragma omp metadirective \
    when(user={condition(sizeof((char){0}) > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(sizeof((short){0}) > 0)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @identical_compound_types(
// CHECK-NOT: @__kmpc_taskyield
// CHECK: call {{.*}} @__kmpc_taskwait
// CHECK-NOT: @__kmpc_taskyield
// CHECK: ret void
void identical_compound_types(void) {
#pragma omp metadirective \
    when(user={condition(sizeof((short){0}) > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(sizeof((short){0}) > 0)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @distinct_va_arg_types(
// CHECK-NOT: @__kmpc_taskwait
// CHECK: call {{.*}} @__kmpc_taskyield
// CHECK-NOT: @__kmpc_taskwait
// CHECK: ret void
void distinct_va_arg_types(void) {
  __builtin_va_list ap;
#pragma omp metadirective \
    when(user={condition(sizeof(__builtin_va_arg(ap, int)) > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(sizeof(__builtin_va_arg(ap, long)) > 0)}, \
         device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @identical_va_arg_types(
// CHECK-NOT: @__kmpc_taskyield
// CHECK: call {{.*}} @__kmpc_taskwait
// CHECK-NOT: @__kmpc_taskyield
// CHECK: ret void
void identical_va_arg_types(void) {
  __builtin_va_list ap;
#pragma omp metadirective \
    when(user={condition(sizeof(__builtin_va_arg(ap, int)) > 0)}, \
         implementation={vendor(score(4): llvm)}: taskyield) \
    when(user={condition(sizeof(__builtin_va_arg(ap, int)) > 0)}, \
         device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}

// CHECK-LABEL: define {{.*}} @higher_score(
// CHECK-NOT: @__kmpc_taskwait
// CHECK: call {{.*}} @__kmpc_taskyield
// CHECK-NOT: @__kmpc_taskwait
// CHECK: ret void
void higher_score(void) {
#pragma omp metadirective \
    when(user={condition(1)}, \
         implementation={vendor(score(5): llvm)}: taskyield) \
    when(user={condition(1)}, device={kind(cpu)}, \
         implementation={vendor(score(3): llvm)}: taskwait)
}
