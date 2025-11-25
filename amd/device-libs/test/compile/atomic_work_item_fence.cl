// Check that the cl_mem_fence_flags is honored.

// GCN:      @test_local()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel, !mmra ![[LOCAL_MMRA:[0-9]+]]
// GCN-NEXT:   ret void
kernel void test_local() {
    atomic_work_item_fence(CLK_LOCAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
}

// GCN:      @test_image()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel, !mmra ![[GLOBAL_MMRA:[0-9]+]]
// GCN-NEXT:   ret void
kernel void test_image() {
    atomic_work_item_fence(CLK_IMAGE_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
}

// GCN:      @test_global()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel, !mmra ![[GLOBAL_MMRA:[0-9]+]]
// GCN-NEXT:   ret void
kernel void test_global() {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
}

// GCN:      @test_local_global()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel{{$}}
// GCN-NEXT:   ret void
kernel void test_local_global() {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
}

// GCN:      @test_all()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel{{$}}
// GCN-NEXT:   ret void
kernel void test_all() {
    atomic_work_item_fence(CLK_IMAGE_MEM_FENCE | CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
}

// GCN:      @test_invalid()
// GCN-NEXT: entry:
// GCN-NEXT:   fence syncscope("agent") acq_rel{{$}}
// GCN-NEXT:   ret void
kernel void test_invalid() {
    atomic_work_item_fence(0, memory_order_acq_rel, memory_scope_device);
}

// GCN: ![[LOCAL_MMRA]]  = !{!"amdgpu-as", !"local"}
// GCN: ![[GLOBAL_MMRA]] = !{!"amdgpu-as", !"global"}
