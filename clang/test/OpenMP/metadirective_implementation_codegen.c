// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple aarch64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple ppc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void bar(void);

void foo(void) {
#pragma omp metadirective when(implementation = {vendor(score(0)  \
                                                        : amd)}, \
                               device = {kind(cpu)}               \
                               : parallel) default(target teams)
  bar();
#pragma omp metadirective when(device = {kind(gpu)}                                 \
                               : target teams) when(implementation = {vendor(amd)} \
                                                    : parallel) default()
  bar();
#pragma omp metadirective default(target) when(implementation = {vendor(score(5)  \
                                                                        : amd)}, \
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

#endif
