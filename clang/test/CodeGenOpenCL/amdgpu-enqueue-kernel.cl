// RUN: %clang_cc1 %s -cl-std=CL2.0 -O0 -emit-llvm -o - -triple amdgcn | FileCheck %s --check-prefix=CHECK

typedef struct {int a;} ndrange_t;

// CHECK-LABEL: define amdgpu_kernel void @test
kernel void test(global char *a, char b, global long *c, long d) {
  queue_t default_queue;
  unsigned flags = 0;
  ndrange_t ndrange;

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                 a[0] = b;
                 });

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(void) {
                 a[0] = b;
                 c[0] = d;
                 });
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *lp) {
                 a[0] = b;
                 c[0] = d;
                 ((local int*)lp)[0] = 1;
                 }, 100);
}

// CHECK-LABEL: define internal amdgpu_kernel void @__test_block_invoke_kernel(<{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i8 }>)
// CHECK-SAME: #[[ATTR:[0-9]+]] !kernel_arg_addr_space !{{.*}} !kernel_arg_access_qual !{{.*}} !kernel_arg_type !{{.*}} !kernel_arg_base_type !{{.*}} !kernel_arg_type_qual !{{.*}}
// CHECK: entry:
// CHECK:  %1 = alloca <{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i8 }>, align 8
// CHECK:  store <{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i8 }> %0, <{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i8 }>* %1, align 8
// CHECK:  %2 = addrspacecast <{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i8 }>* %1 to i8 addrspace(4)*
// CHECK:  call void @__test_block_invoke(i8 addrspace(4)* %2)
// CHECK:  ret void
// CHECK:}

// CHECK-LABEL: define internal amdgpu_kernel void @__test_block_invoke_2_kernel(<{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>)
// CHECK-SAME: #[[ATTR]] !kernel_arg_addr_space !{{.*}} !kernel_arg_access_qual !{{.*}} !kernel_arg_type !{{.*}} !kernel_arg_base_type !{{.*}} !kernel_arg_type_qual !{{.*}}

// CHECK-LABEL: define internal amdgpu_kernel void @__test_block_invoke_3_kernel(<{ i32, i32, i8 addrspace(4)*, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, i8 addrspace(3)*)
// CHECK-SAME: #[[ATTR]] !kernel_arg_addr_space !{{.*}} !kernel_arg_access_qual !{{.*}} !kernel_arg_type !{{.*}} !kernel_arg_base_type !{{.*}} !kernel_arg_type_qual !{{.*}}

// CHECK: attributes #[[ATTR]] = { nounwind "enqueued-block" }
