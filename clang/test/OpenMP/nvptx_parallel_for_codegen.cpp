// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template<typename tx>
tx ftemplate(int n) {
  tx b[10];

  #pragma omp target
  {
    tx d = n;
    #pragma omp parallel for
    for(int i=0; i<10; i++) {
      b[i] += d;
    }
    b[3] += 1;
  }

  return b[3];
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// CHECK: [[MEM_TY:%.+]] = type { [128 x i8] }
// CHECK-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// CHECK-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// CHECK-DAG: [[KERNEL_SIZE:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// CHECK-DAG: [[KERNEL_SHARED:@.+]] = internal unnamed_addr constant i16 1

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l12}}_worker()
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call i1 @__kmpc_kernel_parallel(
// CHECK: call void @__omp_outlined___wrapper(

// CHECK: define weak void @__omp_offloading_{{.*}}l12(
// CHECK: call void @__omp_offloading_{{.*}}l12_worker()
// CHECK: call void @__kmpc_kernel_init(
// CHECK: call void @__kmpc_data_sharing_init_stack()
// CHECK: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// CHECK: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE]],
// CHECK: call void @__kmpc_get_team_static_memory(i16 0, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i64 %7, i16 %6, i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CHECK: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CHECK: [[STACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// CHECK: call void @__kmpc_kernel_prepare_parallel(
// CHECK: call void @__kmpc_begin_sharing_variables({{.*}}, i64 2)
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_end_sharing_variables()
// CHECK: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// CHECK: call void @__kmpc_restore_team_static_memory(i16 0, i16 [[IS_SHARED]])
// CHECK: call void @__kmpc_kernel_deinit(i16 1)

// CHECK: define internal void @__omp_outlined__(
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: [[OMP_IV:%.*]] = alloca i32
// CHECK: store i32 0, {{.*}} [[OMP_LB:%.+]],
// CHECK: store i32 9, {{.*}} [[OMP_UB:%.+]],
// CHECK: store i32 1, {{.*}} [[OMP_ST:%.+]],
// CHECK: call void @__kmpc_for_static_init_4({{.*}} i32 33, {{.*}} [[OMP_LB]], {{.*}} [[OMP_UB]], {{.*}} [[OMP_ST]], i32 1, i32 1)
// CHECK: br label %[[OMP_DISPATCH_COND:.+]]

// CHECK: [[OMP_DISPATCH_COND]]
// CHECK: [[OMP_UB_1:%.+]] = load {{.*}} [[OMP_UB]]
// CHECK: [[COMP_1:%.+]] = icmp sgt {{.*}} [[OMP_UB_1]]
// CHECK: br i1 [[COMP_1]], label %[[COND_TRUE:.+]], label %[[COND_FALSE:.+]]

// CHECK: [[COND_TRUE]]
// CHECK: br label %[[COND_END:.+]]

// CHECK: [[COND_FALSE]]
// CHECK: [[OMP_UB_2:%.+]] = load {{.*}}* [[OMP_UB]]
// CHECK: br label %[[COND_END]]

// CHECK: [[COND_END]]
// CHECK: [[COND_RES:%.+]] = phi i32 [ 9, %[[COND_TRUE]] ], [ [[OMP_UB_2]], %[[COND_FALSE]] ]
// CHECK: store i32 [[COND_RES]], i32* [[OMP_UB]]
// CHECK: [[OMP_LB_1:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK: store i32 [[OMP_LB_1]], i32* [[OMP_IV]]
// CHECK: [[OMP_IV_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[OMP_UB_3:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK: [[COMP_2:%.+]] = icmp sle i32 [[OMP_IV_1]], [[OMP_UB_3]]
// CHECK: br i1 [[COMP_2]], label %[[DISPATCH_BODY:.+]], label %[[DISPATCH_END:.+]]

// CHECK: [[DISPATCH_BODY]]
// CHECK: br label %[[OMP_INNER_FOR_COND:.+]]

// CHECK: [[OMP_INNER_FOR_COND]]
// CHECK: [[OMP_IV_2:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[OMP_UB_4:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK: [[COMP_3:%.+]] = icmp sle i32 [[OMP_IV_2]], [[OMP_UB_4]]
// CHECK: br i1 [[COMP_3]], label %[[OMP_INNER_FOR_BODY:.+]], label %[[OMP_INNER_FOR_END:.+]]

// CHECK: [[OMP_INNER_FOR_BODY]]
// CHECK: br label %[[OMP_BODY_CONTINUE:.+]]

// CHECK: [[OMP_BODY_CONTINUE]]
// CHECK: br label %[[OMP_INNER_FOR_INC:.+]]

// CHECK: [[OMP_INNER_FOR_INC]]
// CHECK: [[OMP_IV_3:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[ADD_1:%.+]] = add nsw i32 [[OMP_IV_3]], 1
// CHECK: store i32 [[ADD_1]], i32* [[OMP_IV]]
// CHECK: br label %[[OMP_INNER_FOR_COND]]

// CHECK: [[OMP_INNER_FOR_COND]]
// CHECK: br label %[[OMP_DISPATCH_INC:.+]]

// CHECK: [[OMP_DISPATCH_INC]]
// CHECK: [[OMP_LB_2:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK: [[OMP_ST_1:%.+]] = load i32, i32* [[OMP_ST]]
// CHECK: [[ADD_2:%.+]] = add nsw i32 [[OMP_LB_2]], [[OMP_ST_1]]
// CHECK: store i32 [[ADD_2]], i32* [[OMP_LB]]
// CHECK: [[OMP_UB_5:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK: [[OMP_ST_2:%.+]] = load i32, i32* [[OMP_ST]]
// CHECK: [[ADD_3:%.+]] = add nsw i32 [[OMP_UB_5]], [[OMP_ST_2]]
// CHECK: store i32 [[ADD_3]], i32* [[OMP_UB]]

// CHECK: [[DISPATCH_END]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

#endif
