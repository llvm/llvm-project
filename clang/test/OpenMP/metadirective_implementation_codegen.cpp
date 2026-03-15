// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void bar();

void foo() {
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

// CHECK-LABEL: void @_Z3foov()
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_2:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_3:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_4:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_5:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_6:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_7:@.+]])
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_2]](
// CHECK: @_Z3barv
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_3]](
// CHECK: @_Z3barv
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_4]](
// CHECK: @_Z3barv
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_5]](
// NO-CHECK: call void @__kmpc_for_static_init
// NO-CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_6]](
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_7]](
// NO-CHECK: call void @__kmpc_for_static_init
// NO-CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

#endif
