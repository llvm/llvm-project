// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

int a;

int foo(int *a);

int main(int argc, char **argv) {
  int b[10], c[10], d[10];
#pragma omp target teams map(tofrom:a)
#pragma omp distribute parallel for firstprivate(b) lastprivate(c) if(a)
  for (int i= 0; i < argc; ++i)
    a = foo(&i) + foo(&a) + foo(&b[i]) + foo(&c[i]) + foo(&d[i]);
  return 0;
}

// CHECK: @__omp_offloading_{{.*}}_main_l17_exec_mode = weak constant i8 1

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_main_l17_worker(

// CHECK: define weak void @__omp_offloading_{{.*}}_main_l17([10 x i32]* dereferenceable(40) %{{.+}}, [10 x i32]* dereferenceable(40) %{{.+}}, i32* dereferenceable(4) %{{.+}}, i{{64|32}} %{{.+}}, [10 x i32]* dereferenceable(40) %{{.+}})
// CHECK: [[PTR:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i{{64|32}} 84, i16 0)
// CHECK: [[STACK:%.+]] = bitcast i8* [[PTR]] to %struct._globalized_locals_ty*
// CHECK: [[ARGC:%.+]] = load i32, i32* %{{.+}}, align
// CHECK: [[ARGC_ADDR:%.+]] = getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[STACK]], i{{32|64}} 0, i{{32|64}} 0
// CHECK: store i32 [[ARGC]], i32* [[ARGC_ADDR]],
// CHECK: getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[STACK]], i{{32|64}} 0, i{{32|64}} 1
// CHECK: getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[STACK]], i{{32|64}} 0, i{{32|64}} 2
// CHECK: call void @__kmpc_for_static_init_4(

// CHECK: call void @__kmpc_serialized_parallel(
// CHECK: call void [[PARALLEL:@.+]](
// CHECK: call void @__kmpc_end_serialized_parallel(

// CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @

// CHECK: call void @__kmpc_data_sharing_pop_stack(i8* [[PTR]])

// CHECK: define internal void [[PARALLEL]](
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack(

// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(

#endif
