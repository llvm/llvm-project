// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple aarch64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple ppc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void bar(void);

void foo(void) {
#pragma omp metadirective when(implementation = {vendor(score(0)  \
                                                        : llvm)}, \
                               device = {kind(cpu)}               \
                               : parallel) default(target teams)
  bar();
#pragma omp metadirective when(device = {kind(gpu)}                                 \
                               : target teams) when(implementation = {vendor(llvm)} \
                                                    : parallel) default()
  bar();
#pragma omp metadirective default(target) when(implementation = {vendor(score(5)  \
                                                                        : llvm)}, \
                                               device = {kind(cpu, host)}         \
                                               : parallel)
  bar();
#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_none)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
}

// CHECK: void @foo()
// CHECK-COUNT-6: ...) @__kmpc_fork_call(
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined(
// CHECK: @bar
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined.1(
// CHECK: @bar
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined.2(
// CHECK: @bar
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined.3(
// NO-CHECK: call void @__kmpc_for_static_init
// NO-CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined.4(
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void @foo.omp_outlined.5(
// NO-CHECK: call void @__kmpc_for_static_init
// NO-CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// All three score 5. The first is a strict subset of the third, which ties
// with the second, so the second is selected.
void tie(void) {
#pragma omp metadirective                                                    \
    when(user = {condition(1)}, implementation = {vendor(score(4) : llvm)}   \
         : taskyield)                                                        \
    when(device = {kind(cpu)}, implementation = {vendor(score(3) : llvm)}    \
         : taskwait)                                                         \
    when(user = {condition(1)}, device = {kind(host)},                       \
         implementation = {vendor(score(3) : llvm)} : barrier)
}

// CHECK-LABEL: define {{.*}}void @tie(
// CHECK-NOT: call {{.*}}@__kmpc_omp_taskyield(
// CHECK-NOT: call {{.*}}@__kmpc_barrier(
// CHECK: call {{.*}}@__kmpc_omp_taskwait(
// CHECK-NOT: call {{.*}}@__kmpc_barrier(
// CHECK: ret void

#endif
