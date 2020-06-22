// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s
// expected-no-diagnostics

int foo(int &a) { return a; }

int bar() {
  int a;
  return foo(a);
}

// CHECK: define weak void @__omp_offloading_{{.*}}maini1{{.*}}_l[[@LINE+5]](i32* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %{{.*}})
// CHECK-NOT: @__kmpc_data_sharing_coalesced_push_stack

int maini1() {
  int a;
#pragma omp target parallel map(from:a)
  {
    int b;
    a = foo(b) + bar();
  }
  return a;
}

// parallel region
// CHECK: define {{.*}}void @{{.*}}(i32* noalias {{.*}}, i32* noalias {{.*}}, i32* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %{{.*}})
// CHECK-NOT: call i8* @__kmpc_data_sharing_coalesced_push_stack(
// CHECK: [[B_ADDR:%.+]] = alloca i32,
// CHECK: call {{.*}}[[FOO:@.*foo.*]](i32* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[B_ADDR]])
// CHECK: call {{.*}}[[BAR:@.*bar.*]]()
// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(
// CHECK: ret void

// CHECK: define {{.*}}[[FOO]](i32* nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK-NOT: @__kmpc_data_sharing_coalesced_push_stack

// CHECK: define {{.*}}[[BAR]]()
// CHECK: alloca i32,
// CHECK: [[A_LOCAL_ADDR:%.+]] = alloca i32,
// CHECK: [[RES:%.+]] = call i8 @__kmpc_is_spmd_exec_mode()
// CHECK: [[IS_SPMD:%.+]] = icmp ne i8 [[RES]], 0
// CHECK: br i1 [[IS_SPMD]], label
// CHECK: br label
// CHECK: [[RES:%.+]] = call i8* @__kmpc_data_sharing_coalesced_push_stack(i64 128, i16 0)
// CHECK: [[GLOBALS:%.+]] = bitcast i8* [[RES]] to [[GLOBAL_ST:%.+]]*
// CHECK: br label
// CHECK: [[ITEMS:%.+]] = phi [[GLOBAL_ST]]* [ null, {{.+}} ], [ [[GLOBALS]], {{.+}} ]
// CHECK: [[A_ADDR:%.+]] = getelementptr inbounds [[GLOBAL_ST]], [[GLOBAL_ST]]* [[ITEMS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK: [[LID:%.+]] = and i32 [[TID]], 31
// CHECK: [[A_GLOBAL_ADDR:%.+]] = getelementptr inbounds [32 x i32], [32 x i32]* [[A_ADDR]], i32 0, i32 [[LID]]
// CHECK: [[A_ADDR:%.+]] = select i1 [[IS_SPMD]], i32* [[A_LOCAL_ADDR]], i32* [[A_GLOBAL_ADDR]]
// CHECK: call {{.*}}[[FOO]](i32* nonnull align {{[0-9]+}} dereferenceable{{.*}} [[A_ADDR]])
// CHECK: br i1 [[IS_SPMD]], label
// CHECK: [[BC:%.+]] = bitcast [[GLOBAL_ST]]* [[ITEMS]] to i8*
// CHECK: call void @__kmpc_data_sharing_pop_stack(i8* [[BC]])
// CHECK: br label
// CHECK: ret i32
