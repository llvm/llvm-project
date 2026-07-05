// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple aarch64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple ppc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void bar(void);

void foo(void) {
#pragma omp metadirective when(device = {kind(any)} \
                               : parallel)
  bar();
#pragma omp metadirective when(device = {kind(host, cpu)} \
                               : parallel for num_threads(4))
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(device = {kind(host)} \
                               : parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(device = {kind(nohost, gpu)} \
                               :) when(device = {kind(cpu)} \
                                       : parallel)
  bar();
#pragma omp metadirective when(device = {kind(any, cpu)} \
                               : parallel)
  bar();
#pragma omp metadirective when(device = {kind(any, host)} \
                               : parallel)
  bar();
#pragma omp metadirective when(device = {kind(gpu)} \
                               : target parallel for) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
}

// CHECK-LABEL: define {{.+}} void @foo()
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_1:@.+]])
// CHECK-NEXT: @__kmpc_push_num_threads
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_2:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_3:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_4:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_5:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_6:@.+]])
// CHECK: @__kmpc_fork_call(ptr {{.+}}, i32 0, ptr [[OUTLINED_7:@.+]])
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_1]](
// CHECK: call void @bar
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_2]](
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_3]](
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_4]](
// CHECK: call void @bar
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_5]](
// CHECK: call void @bar
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_6]](
// CHECK: call void @bar
// CHECK: ret void

// CHECK: define internal void [[OUTLINED_7]](
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void

#endif
