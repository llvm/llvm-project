// Make sure that invoking blocks in static functions with the same name in
// different modules are linked together.

// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -fno-ident -DKERNEL_NAME=test_kernel_first -DTYPE=float -DCONST=256.0f -emit-llvm-bc -o %t.0.bc %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -fno-ident -DKERNEL_NAME=test_kernel_second -DTYPE=int -DCONST=128.0f -emit-llvm-bc -o %t.1.bc %s

// Make sure nothing strange happens with the linkage choices.
// RUN: opt -passes=globalopt -o %t.opt.0.bc %t.0.bc
// RUN: opt -passes=globalopt -o %t.opt.1.bc %t.1.bc

// Check the result of linking
// RUN: llvm-link -S %t.opt.0.bc %t.opt.1.bc -o - | FileCheck %s

// Make sure that a block invoke used with the same name works in multiple
// translation units

// CHECK: @llvm.used = appending addrspace(1) global [4 x ptr] [ptr @__static_invoker_block_invoke_kernel, ptr addrspacecast (ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle to ptr), ptr @__static_invoker_block_invoke_kernel.2, ptr addrspacecast (ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle.3 to ptr)], section "llvm.metadata"


// CHECK: @__static_invoker_block_invoke_kernel.runtime.handle = internal addrspace(1) externally_initialized constant %block.runtime.handle.t zeroinitializer, section ".amdgpu.kernel.runtime.handle"
// CHECK: @__static_invoker_block_invoke_kernel.runtime.handle.3 = internal addrspace(1) externally_initialized constant %block.runtime.handle.t zeroinitializer, section ".amdgpu.kernel.runtime.handle"

// CHECK: define internal amdgpu_kernel void @__static_invoker_block_invoke_kernel(<{ i32, i32, ptr, ptr addrspace(1), ptr addrspace(1) }> %0) #{{[0-9]+}} !associated ![[ASSOC_FIRST_MD:[0-9]+]]


// CHECK-LABEL: define internal void @__static_invoker_block_invoke(ptr noundef %.block_descriptor)
// CHECK: call float @llvm.fmuladd.f32


// CHECK-LABEL: define dso_local amdgpu_kernel void @test_kernel_first(


// CHECK-LABEL: define internal fastcc void @static_invoker(ptr addrspace(1) noundef %outptr, ptr addrspace(1) noundef %argptr)
// CHECK:  call i32 @__enqueue_kernel_basic(ptr addrspace(1) %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr addrspace(5) %tmp, ptr addrspacecast (ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle to ptr), ptr %{{.+}})

// CHECK: declare i32 @__enqueue_kernel_basic(ptr addrspace(1), i32, ptr addrspace(5), ptr, ptr) local_unnamed_addr


// CHECK: define internal amdgpu_kernel void @__static_invoker_block_invoke_kernel.2(<{ i32, i32, ptr, ptr addrspace(1), ptr addrspace(1) }> %0) #{{[0-9]+}} !associated ![[ASSOC_SECOND_MD:[0-9]+]]
// CHECK: call void @__static_invoker_block_invoke.4(ptr %


// CHECK-LABEL: define internal void @__static_invoker_block_invoke.4(ptr noundef %.block_descriptor)
// CHECK: mul nsw i32
// CHECK: sitofp
// CHECK: fadd
// CHECK: fptosi

// CHECK-LABEL: define dso_local amdgpu_kernel void @test_kernel_second(ptr addrspace(1) noundef align 4 %outptr, ptr addrspace(1) noundef align 4 %argptr, ptr addrspace(1) noundef align 4 %difference)

// CHECK-LABEL: define internal fastcc void @static_invoker.5(ptr addrspace(1) noundef %outptr, ptr addrspace(1) noundef %argptr) unnamed_addr #{{[0-9]+}} {
// CHECK: call i32 @__enqueue_kernel_basic(ptr addrspace(1) %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr addrspace(5) %tmp, ptr addrspacecast (ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle.3 to ptr), ptr %{{.+}})


typedef struct {int a;} ndrange_t;

static void static_invoker(global TYPE* outptr, global TYPE* argptr) {
  queue_t default_queue;
  unsigned flags = 0;
  ndrange_t ndrange;

  enqueue_kernel(default_queue, flags, ndrange,
  ^(void) {
  global TYPE* f = argptr;
  outptr[0] = f[1] * f[2] + CONST;
  });
}

kernel void KERNEL_NAME(global TYPE *outptr, global TYPE *argptr, global TYPE *difference) {
  queue_t default_queue;
  unsigned flags = 0;
  ndrange_t ndrange;

  static_invoker(outptr, argptr);

  *difference = CONST;
}

// CHECK: ![[ASSOC_FIRST_MD]] = !{ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle}
// CHECK: ![[ASSOC_SECOND_MD]] = !{ptr addrspace(1) @__static_invoker_block_invoke_kernel.runtime.handle.3}
