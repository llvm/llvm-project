// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

int foo(int &a) { return a; }

int bar() {
  int a;
  return foo(a);
}

// CHECK: define weak void @__omp_offloading_{{.*}}maini1{{.*}}_l[[@LINE+5]](i32* dereferenceable{{.*}})
// CHECK-NOT: @__kmpc_data_sharing_push_stack

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
// CHECK: define {{.*}}void @{{.*}}(i32* noalias {{.*}}, i32* noalias {{.*}}, i32* dereferenceable{{.*}})
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack(
// CHECK: [[B_ADDR:%.+]] = alloca i32,
// CHECK: call {{.*}}[[FOO:@.*foo.*]](i32* dereferenceable{{.*}} [[B_ADDR]])
// CHECK: call {{.*}}[[BAR:@.*bar.*]]()
// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(
// CHECK: ret void

// CHECK: define {{.*}}[[FOO]](i32* dereferenceable{{.*}})
// CHECK-NOT: @__kmpc_data_sharing_push_stack

// CHECK: define {{.*}}[[BAR]]()
// CHECK: [[STACK:%.+]] = alloca [[GLOBAL_ST:%.+]],
// CHECK: [[RES:%.+]] = call i8 @__kmpc_is_spmd_exec_mode()
// CHECK: [[IS_SPMD:%.+]] = icmp ne i8 [[RES]], 0
// CHECK: br i1 [[IS_SPMD]], label
// CHECK: br label
// CHECK: [[RES:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i64 4, i16 0)
// CHECK: [[GLOBALS:%.+]] = bitcast i8* [[RES]] to [[GLOBAL_ST]]*
// CHECK: br label
// CHECK: [[ITEMS:%.+]] = phi [[GLOBAL_ST]]* [ [[STACK]], {{.+}} ], [ [[GLOBALS]], {{.+}} ]
// CHECK: [[A_ADDR:%.+]] = getelementptr inbounds [[GLOBAL_ST]], [[GLOBAL_ST]]* [[ITEMS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: call {{.*}}[[FOO]](i32* dereferenceable{{.*}} [[A_ADDR]])
// CHECK: br i1 [[IS_SPMD]], label
// CHECK: [[BC:%.+]] = bitcast [[GLOBAL_ST]]* [[ITEMS]] to i8*
// CHECK: call void @__kmpc_data_sharing_pop_stack(i8* [[BC]])
// CHECK: br label
// CHECK: ret i32
