// Test device global memory data sharing codegen.
///==========================================================================///

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK1

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void test_ds(){
  #pragma omp target
  {
    int a = 10;
    #pragma omp parallel
    {
      a = 1000;
    }
    int b = 100;
    int c = 1000;
    #pragma omp parallel private(c)
    {
      int *c1 = &c;
      b = a + 10000;
    }
  }
}

/// ========= In the worker function ========= ///
// CK1: {{.*}}define internal void @__omp_offloading{{.*}}test_ds{{.*}}_worker()
// CK1: call void @llvm.nvvm.barrier0()
// CK1: call void @__kmpc_data_sharing_init_stack

/// ========= In the kernel function ========= ///

// CK1: {{.*}}define void @__omp_offloading{{.*}}test_ds{{.*}}()
// CK1: [[SHAREDARGS1:%.+]] = alloca i8**
// CK1: [[SHAREDARGS2:%.+]] = alloca i8**
// CK1: call void @__kmpc_kernel_init
// CK1: call void @__kmpc_data_sharing_init_stack
// CK1: [[GLOBALSTACK:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i64 8, i16 0)
// CK1: [[GLOBALSTACK2:%.+]] = bitcast i8* [[GLOBALSTACK]] to %struct._globalized_locals_ty*
// CK1: [[A:%.+]] = getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[GLOBALSTACK2]], i32 0, i32 0
// CK1: [[B:%.+]] = getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[GLOBALSTACK2]], i32 0, i32 1
// CK1: store i32 10, i32* [[A]]
// CK1: call void @__kmpc_kernel_prepare_parallel({{.*}}, i16 1)
// CK1: call void @__kmpc_begin_sharing_variables(i8*** [[SHAREDARGS1]], i64 1)
// CK1: [[SHARGSTMP1:%.+]] = load i8**, i8*** [[SHAREDARGS1]]
// CK1: [[SHARGSTMP2:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP1]], i64 0
// CK1: [[SHAREDVAR:%.+]] = bitcast i32* [[A]] to i8*
// CK1: store i8* [[SHAREDVAR]], i8** [[SHARGSTMP2]]
// CK1: call void @llvm.nvvm.barrier0()
// CK1: call void @llvm.nvvm.barrier0()
// CK1: call void @__kmpc_end_sharing_variables()
// CK1: store i32 100, i32* [[B]]
// CK1: call void @__kmpc_kernel_prepare_parallel({{.*}}, i16 1)
// CK1: call void @__kmpc_begin_sharing_variables(i8*** [[SHAREDARGS2]], i64 2)
// CK1: [[SHARGSTMP3:%.+]] = load i8**, i8*** [[SHAREDARGS2]]
// CK1: [[SHARGSTMP4:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP3]], i64 0
// CK1: [[SHAREDVAR1:%.+]] = bitcast i32* [[B]] to i8*
// CK1: store i8* [[SHAREDVAR1]], i8** [[SHARGSTMP4]]
// CK1: [[SHARGSTMP12:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP3]], i64 1
// CK1: [[SHAREDVAR2:%.+]] = bitcast i32* [[A]] to i8*
// CK1: store i8* [[SHAREDVAR2]], i8** [[SHARGSTMP12]]
// CK1: call void @llvm.nvvm.barrier0()
// CK1: call void @llvm.nvvm.barrier0()
// CK1: call void @__kmpc_end_sharing_variables()
// CK1: call void @__kmpc_data_sharing_pop_stack(i8* [[GLOBALSTACK]])
// CK1: call void @__kmpc_kernel_deinit(i16 1)

/// ========= In the data sharing wrapper function ========= ///

// CK1: {{.*}}define internal void @__omp_outlined{{.*}}wrapper({{.*}})
// CK1: [[SHAREDARGS4:%.+]] = alloca i8**
// CK1: call void @__kmpc_get_shared_variables(i8*** [[SHAREDARGS4]])
// CK1: [[SHARGSTMP13:%.+]] = load i8**, i8*** [[SHAREDARGS4]]
// CK1: [[SHARGSTMP14:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP13]], i64 0
// CK1: [[SHARGSTMP15:%.+]] = bitcast i8** [[SHARGSTMP14]] to i32**
// CK1: [[SHARGSTMP16:%.+]] = load i32*, i32** [[SHARGSTMP15]]
// CK1: call void @__omp_outlined__{{.*}}({{.*}}, i32* [[SHARGSTMP16]])

/// outlined function for the second parallel region ///

// CK1: define internal void @{{.+}}(i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable{{.+}}, i32* dereferenceable{{.+}})
// CK1: [[RES:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i64 4, i16 0)
// CK1: [[GLOBALS:%.+]] = bitcast i8* [[RES]] to [[GLOBAL_TY:%.+]]*
// CK1: [[C_ADDR:%.+]] = getelementptr inbounds [[GLOBAL_TY]], [[GLOBAL_TY]]* [[GLOBALS]], i32 0, i32 0
// CK1: store i32* [[C_ADDR]], i32** %
// CK1: call void @__kmpc_data_sharing_pop_stack(i8* [[RES]])

/// ========= In the data sharing wrapper function ========= ///

// CK1: {{.*}}define internal void @__omp_outlined{{.*}}wrapper({{.*}})
// CK1: [[SHAREDARGS3:%.+]] = alloca i8**
// CK1: call void @__kmpc_get_shared_variables(i8*** [[SHAREDARGS3]])
// CK1: [[SHARGSTMP5:%.+]] = load i8**, i8*** [[SHAREDARGS3]]
// CK1: [[SHARGSTMP6:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP5]], i64 0
// CK1: [[SHARGSTMP7:%.+]] = bitcast i8** [[SHARGSTMP6]] to i32**
// CK1: [[SHARGSTMP8:%.+]] = load i32*, i32** [[SHARGSTMP7]]
// CK1: [[SHARGSTMP9:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP5]], i64 1
// CK1: [[SHARGSTMP10:%.+]] = bitcast i8** [[SHARGSTMP9]] to i32**
// CK1: [[SHARGSTMP11:%.+]] = load i32*, i32** [[SHARGSTMP10]]
// CK1: call void @__omp_outlined__{{.*}}({{.*}}, i32* [[SHARGSTMP8]], i32* [[SHARGSTMP11]])

#endif

