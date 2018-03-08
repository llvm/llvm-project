// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=COMMON,AMDGCN %s

// COMMON: @__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }
// COMMON-NOT: .str

// SPIR-LABEL: define internal {{.*}}void @block_A_block_invoke(i8 addrspace(4)* %.block_descriptor, i8 addrspace(3)* %a)
// AMDGCN-LABEL: define internal {{.*}}void @block_A_block_invoke(i8* %.block_descriptor, i8 addrspace(3)* %a)
void (^block_A)(local void *) = ^(local void *a) {
  return;
};

// COMMON-LABEL: define {{.*}}void @foo()
void foo(){
  int i;
  // COMMON-NOT: %block.isa
  // COMMON-NOT: %block.flags
  // COMMON-NOT: %block.reserved
  // COMMON-NOT: %block.descriptor
  // SPIR: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }>* %[[block:.*]], i32 0, i32 0
  // AMDGCN: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }> addrspace(5)* %[[block:.*]], i32 0, i32 0
  // SPIR: store i32 12, i32* %[[block_size]]
  // AMDGCN: store i32 12, i32 addrspace(5)* %[[block_size]]
  // SPIR: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }>* %[[block]], i32 0, i32 1
  // AMDGCN: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }> addrspace(5)* %[[block]], i32 0, i32 1
  // SPIR: store i32 4, i32* %[[block_align]]
  // AMDGCN: store i32 4, i32 addrspace(5)* %[[block_align]]
  // SPIR: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }>* %[[block]], i32 0, i32 2
  // SPIR: %[[i_value:.*]] = load i32, i32* %i
  // SPIR: store i32 %[[i_value]], i32* %[[block_captured]],
  // SPIR: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i32 }>* %[[block]] to i32 ()*
  // SPIR: %[[blk_gen_ptr:.*]] = addrspacecast i32 ()* %[[blk_ptr]] to i32 () addrspace(4)*
  // SPIR: store i32 () addrspace(4)* %[[blk_gen_ptr]], i32 () addrspace(4)** %[[block_B:.*]],
  // SPIR: %[[block_literal:.*]] = load i32 () addrspace(4)*, i32 () addrspace(4)** %[[block_B]]
  // SPIR: %[[blk_gen_ptr:.*]] = bitcast i32 () addrspace(4)* %[[block_literal]] to i8 addrspace(4)*
  // SPIR: call {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* %[[blk_gen_ptr]])
  // AMDGCN: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }> addrspace(5)* %[[block]], i32 0, i32 2
  // AMDGCN: %[[i_value:.*]] = load i32, i32 addrspace(5)* %i
  // AMDGCN: store i32 %[[i_value]], i32 addrspace(5)* %[[block_captured]],
  // AMDGCN: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i32 }> addrspace(5)* %[[block]] to i32 () addrspace(5)*
  // AMDGCN: %[[blk_gen_ptr:.*]] = addrspacecast i32 () addrspace(5)* %[[blk_ptr]] to i32 ()*
  // AMDGCN: store i32 ()* %[[blk_gen_ptr]], i32 ()* addrspace(5)* %[[block_B:.*]],
  // AMDGCN: %[[block_literal:.*]] = load i32 ()*, i32 ()* addrspace(5)* %[[block_B]]
  // AMDGCN: %[[blk_gen_ptr:.*]] = bitcast i32 ()* %[[block_literal]] to i8*
  // AMDGCN: call {{.*}}i32 @__foo_block_invoke(i8* %[[blk_gen_ptr]])

  int (^ block_B)(void) = ^{
    return i;
  };
  block_B();
}

// SPIR-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor)
// SPIR:  %[[block:.*]] = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i32 }> addrspace(4)*
// SPIR:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }> addrspace(4)* %[[block]], i32 0, i32 2
// SPIR:  %[[block_capture:.*]] = load i32, i32 addrspace(4)* %[[block_capture_addr]]
// AMDGCN-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8* %.block_descriptor)
// AMDGCN:  %[[block:.*]] = bitcast i8* %.block_descriptor to <{ i32, i32, i32 }>*
// AMDGCN:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i32 }>, <{ i32, i32, i32 }>* %[[block]], i32 0, i32 2
// AMDGCN:  %[[block_capture:.*]] = load i32, i32* %[[block_capture_addr]]

// COMMON-NOT: define{{.*}}@__foo_block_invoke_kernel
