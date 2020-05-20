// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
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

// CHECK: [[MEM_TY:%.+]] = type { [128 x i8] }
// CHECK-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// CHECK-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// CHECK-DAG: [[KERNEL_SIZE:@.+]] = internal unnamed_addr constant i{{64|32}} 40
// CHECK-DAG: [[KERNEL_SHARED:@.+]] = internal unnamed_addr constant i16 1
// CHECK-DAG: @__omp_offloading_{{.*}}_main_l17_exec_mode = weak constant i8 0

// CHECK: define weak void @__omp_offloading_{{.*}}_main_l17([10 x i32]* nonnull align 4 dereferenceable(40) %{{.+}}, [10 x i32]* nonnull align 4 dereferenceable(40) %{{.+}}, i32* nonnull align 4 dereferenceable(4) %{{.+}}, i{{64|32}} %{{.+}}, [10 x i32]* nonnull align 4 dereferenceable(40) %{{.+}})
// CHECK: [[SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// CHECK: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE]],
// CHECK: call void @__kmpc_get_team_static_memory(i16 1, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CHECK: [[PTR:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CHECK: [[GEP:%.+]] = getelementptr inbounds i8, i8* [[PTR]], i{{64|32}} 0
// CHECK: [[STACK:%.+]] = bitcast i8* [[GEP]] to %struct._globalized_locals_ty*
// CHECK: getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[STACK]], i{{32|64}} 0, i{{32|64}} 0
// CHECK-NOT: getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[STACK]],
// CHECK: call void @__kmpc_for_static_init_4(

// CHECK: call void [[PARALLEL:@.+]](

// CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @

// CHECK: [[SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// CHECK: call void @__kmpc_restore_team_static_memory(i16 1, i16 [[SHARED]])

// CHECK: define internal void [[PARALLEL]](
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack(

// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(

#endif
