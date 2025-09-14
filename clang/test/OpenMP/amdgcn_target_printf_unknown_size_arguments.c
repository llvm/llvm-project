// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-is-device -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK
// expected-no-diagnostics

extern int printf(const char *, ...);

int CheckMultipleArgs(int a) {
  char *test = "testing";
  char *t;
#pragma omp target private(t)
  {
    t = test + a;
    printf("%s %d %s", t, 21, test);
// CHECK-LABEL: define weak_odr protected amdgpu_kernel void @{{.*}}CheckMultipleArgs
// CHECK: entry:
// CHECK:   [[DYN_PTR_ADDR:%[a-zA-Z0-9_.]+]] = alloca ptr, align 8, addrspace(5)
// CHECK:   [[TEST_ADDR:%[a-zA-Z0-9_.]+]] = alloca ptr, align 8, addrspace(5)
// CHECK:   [[A_ADDR:%[a-zA-Z0-9_.]+]] = alloca i64, align 8, addrspace(5)
// CHECK:   [[T_ADDR:%[a-zA-Z0-9_.]+]] = alloca ptr, align 8, addrspace(5)
// CHECK:   [[DYN_PTR_CAST:%[a-zA-Z0-9_.]+]] = addrspacecast ptr addrspace(5) [[DYN_PTR_ADDR]] to ptr
// CHECK:   [[TEST_CAST:%[a-zA-Z0-9_.]+]] = addrspacecast ptr addrspace(5) [[TEST_ADDR]] to ptr
// CHECK:   [[A_CAST:%[a-zA-Z0-9_.]+]] = addrspacecast ptr addrspace(5) [[A_ADDR]] to ptr
// CHECK:   [[T_CAST:%[a-zA-Z0-9_.]+]] = addrspacecast ptr addrspace(5) [[T_ADDR]] to ptr
// CHECK:   store ptr %dyn_ptr, ptr [[DYN_PTR_CAST]], align 8
// CHECK:   store ptr %test, ptr [[TEST_CAST]], align 8
// CHECK:   store i64 %a, ptr [[A_CAST]], align 8
// CHECK:   [[INIT_CALL:%[a-zA-Z0-9_.]+]] = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) {{.*}} to ptr), ptr %dyn_ptr)
// CHECK:   [[EXEC_USER_CODE:%[a-zA-Z0-9_.]+]] = icmp eq i32 [[INIT_CALL]], -1
// CHECK:   br i1 [[EXEC_USER_CODE]], label %[[USER_CODE_ENTRY:.+]], label %[[WORKER_EXIT:.+]]

// CHECK: [[USER_CODE_ENTRY]]:
// CHECK:   [[LOAD_TEST:%[0-9]+]] = load ptr, ptr [[TEST_CAST]], align 8
// CHECK:   [[LOAD_A:%[0-9]+]] = load i32, ptr [[A_CAST]], align 4
// CHECK:   %idx.ext = sext i32 [[LOAD_A]] to i64
// CHECK:   %add.ptr = getelementptr inbounds i8, ptr [[LOAD_TEST]], i64 %idx.ext
// CHECK:   store ptr %add.ptr, ptr [[T_CAST]], align 8
// CHECK:   [[LOAD_T:%[0-9]+]] = load ptr, ptr [[T_CAST]], align 8
// CHECK:   [[LOAD_TEST_AGAIN:%[0-9]+]] = load ptr, ptr [[TEST_CAST]], align 8
// CHECK:   call ptr @__llvm_omp_emissary_premalloc(i32 %total_buffer_size)
// CHECK:   call void @__kmpc_target_deinit()
// CHECK:   ret void

// CHECK: [[WORKER_EXIT]]:
// CHECK:   ret void
  }

  return 0;
}
