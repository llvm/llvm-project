// RUN: %clang_cc1 -verify -w -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void bar();

void x86_64_device_isa_selected() {
#pragma omp metadirective when(device = {isa("sse2")} \
                               : parallel) default(single)
  bar();
}
// CHECK-LABEL: void @_Z26x86_64_device_isa_selectedv()
// CHECK: ...) @__kmpc_fork_call{{.*}}@_Z26x86_64_device_isa_selectedv.omp_outlined
// CHECK: ret void

// CHECK: define internal void @_Z26x86_64_device_isa_selectedv.omp_outlined(
// CHECK: @_Z3barv
// CHECK: ret void

void x86_64_device_isa_not_selected() {
#pragma omp metadirective when(device = {isa("some-unsupported-feature")} \
                               : parallel) default(single)
  bar();
}
// CHECK-LABEL: void @_Z30x86_64_device_isa_not_selectedv()
// CHECK: call i32 @__kmpc_single
// CHECK:  @_Z3barv
// CHECK: call void @__kmpc_end_single
// CHECK: ret void

// Both WHEN clauses score the same. The second requires more traits, but not
// the "sse" ISA name, so the first one is not a strict subset and wins.
void x86_64_device_isa_names_differ() {
#pragma omp metadirective when(device = {isa("sse")} : parallel) \
    when(device = {isa("sse2")}, user = {condition(1)} : single)
  bar();
}
// CHECK-LABEL: void @_Z30x86_64_device_isa_names_differv()
// CHECK-NOT: __kmpc_single
// CHECK: ...) @__kmpc_fork_call{{.*}}@_Z30x86_64_device_isa_names_differv.omp_outlined
// CHECK: ret void
#endif
