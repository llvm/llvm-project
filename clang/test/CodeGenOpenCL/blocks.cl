// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa-opencl | FileCheck -check-prefixes=COMMON,AMD %s

// SPIR: %struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }
// AMD: %struct.__opencl_block_literal_generic = type { i32, i32, i8* }
// SPIR: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) to i8 addrspace(4)*) }
// AMD: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8* } { i32 16, i32 8, i8* bitcast (void (i8*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) }
// COMMON-NOT: .str

// SPIR-LABEL: define internal {{.*}}void @block_A_block_invoke(i8 addrspace(4)* %.block_descriptor, i8 addrspace(3)* %a)
// AMD-LABEL: define internal {{.*}}void @block_A_block_invoke(i8* %.block_descriptor, i8 addrspace(3)* %a)
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
  // SPIR: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 0
  // AMD: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 0
  // SPIR: store i32 16, i32* %[[block_size]]
  // AMD: store i32 20, i32 addrspace(5)* %[[block_size]]
  // SPIR: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 1
  // AMD: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 1
  // SPIR: store i32 4, i32* %[[block_align]]
  // AMD: store i32 8, i32 addrspace(5)* %[[block_align]]
  // SPIR: %[[block_invoke:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block:.*]], i32 0, i32 2
  // SPIR: store i8 addrspace(4)* addrspacecast (i8* bitcast (i32 (i8 addrspace(4)*)* @__foo_block_invoke to i8*) to i8 addrspace(4)*), i8 addrspace(4)** %[[block_invoke]]
  // SPIR: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block]], i32 0, i32 3
  // SPIR: %[[i_value:.*]] = load i32, i32* %i
  // SPIR: store i32 %[[i_value]], i32* %[[block_captured]],
  // SPIR: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block]] to i32 ()*
  // SPIR: %[[blk_gen_ptr:.*]] = addrspacecast i32 ()* %[[blk_ptr]] to i32 () addrspace(4)*
  // SPIR: store i32 () addrspace(4)* %[[blk_gen_ptr]], i32 () addrspace(4)** %[[block_B:.*]],
  // SPIR: %[[blk_gen_ptr:.*]] = load i32 () addrspace(4)*, i32 () addrspace(4)** %[[block_B]]
  // SPIR: %[[block_literal:.*]] = bitcast i32 () addrspace(4)* %[[blk_gen_ptr]] to %struct.__opencl_block_literal_generic addrspace(4)*
  // SPIR: %[[invoke_addr:.*]] = getelementptr inbounds %struct.__opencl_block_literal_generic, %struct.__opencl_block_literal_generic addrspace(4)* %[[block_literal]], i32 0, i32 2
  // SPIR: %[[blk_gen_ptr:.*]] = bitcast %struct.__opencl_block_literal_generic addrspace(4)* %[[block_literal]] to i8 addrspace(4)*
  // SPIR: %[[invoke_func_ptr:.*]] = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %[[invoke_addr]]
  // SPIR: %[[invoke_func:.*]] = addrspacecast i8 addrspace(4)* %[[invoke_func_ptr]] to i32 (i8 addrspace(4)*)*
  // SPIR: call {{.*}}i32 %[[invoke_func]](i8 addrspace(4)* %[[blk_gen_ptr]])
  // AMD: %[[block_invoke:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block:.*]], i32 0, i32 2
  // AMD: store i8* bitcast (i32 (i8*)* @__foo_block_invoke to i8*), i8* addrspace(5)* %[[block_invoke]]
  // AMD: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]], i32 0, i32 3
  // AMD: %[[i_value:.*]] = load i32, i32 addrspace(5)* %i
  // AMD: store i32 %[[i_value]], i32 addrspace(5)* %[[block_captured]],
  // AMD: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]] to i32 () addrspace(5)*
  // AMD: %[[blk_gen_ptr:.*]] = addrspacecast i32 () addrspace(5)* %[[blk_ptr]] to i32 ()*
  // AMD: store i32 ()* %[[blk_gen_ptr]], i32 ()* addrspace(5)* %[[block_B:.*]],
  // AMD: %[[blk_gen_ptr:.*]] = load i32 ()*, i32 ()* addrspace(5)* %[[block_B]]
  // AMD: %[[block_literal:.*]] = bitcast i32 ()* %[[blk_gen_ptr]] to %struct.__opencl_block_literal_generic*
  // AMD: %[[invoke_addr:.*]] = getelementptr inbounds %struct.__opencl_block_literal_generic, %struct.__opencl_block_literal_generic* %[[block_literal]], i32 0, i32 2
  // AMD: %[[blk_gen_ptr:.*]] = bitcast %struct.__opencl_block_literal_generic* %[[block_literal]] to i8*
  // AMD: %[[invoke_func_ptr:.*]] = load i8*, i8** %[[invoke_addr]]
  // AMD: %[[invoke_func:.*]] = bitcast i8* %[[invoke_func_ptr]] to i32 (i8*)*
  // AMD: call {{.*}}i32 %[[invoke_func]](i8* %[[blk_gen_ptr]])

  int (^ block_B)(void) = ^{
    return i;
  };
  block_B();
}

// SPIR-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor)
// SPIR:  %[[block:.*]] = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)*
// SPIR:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)* %[[block]], i32 0, i32 3
// SPIR:  %[[block_capture:.*]] = load i32, i32 addrspace(4)* %[[block_capture_addr]]
// AMD-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8* %.block_descriptor)
// AMD:  %[[block:.*]] = bitcast i8* %.block_descriptor to <{ i32, i32, i8*, i32 }>*
// AMD:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }>* %[[block]], i32 0, i32 3
// AMD:  %[[block_capture:.*]] = load i32, i32* %[[block_capture_addr]]

// COMMON-NOT: define{{.*}}@__foo_block_invoke_kernel
