// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=COMMON,AMDGCN %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple spir-unknown-unknown | FileCheck -check-prefixes=CHECK-DEBUG %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=CHECK-DEBUG %s

// SPIR: %struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }
// AMDGCN: %struct.__opencl_block_literal_generic = type { i32, i32, i8* }
// SPIR: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) to i8 addrspace(4)*) }
// AMDGCN: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8* } { i32 16, i32 8, i8* bitcast (void (i8*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) }
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
  // SPIR: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 0
  // AMDGCN: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 0
  // SPIR: store i32 16, i32* %[[block_size]]
  // AMDGCN: store i32 20, i32 addrspace(5)* %[[block_size]]
  // SPIR: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 1
  // AMDGCN: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 1
  // SPIR: store i32 4, i32* %[[block_align]]
  // AMDGCN: store i32 8, i32 addrspace(5)* %[[block_align]]
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
  // AMDGCN: %[[block_invoke:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block:.*]], i32 0, i32 2
  // AMDGCN: store i8* bitcast (i32 (i8*)* @__foo_block_invoke to i8*), i8* addrspace(5)* %[[block_invoke]]
  // AMDGCN: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]], i32 0, i32 3
  // AMDGCN: %[[i_value:.*]] = load i32, i32 addrspace(5)* %i
  // AMDGCN: store i32 %[[i_value]], i32 addrspace(5)* %[[block_captured]],
  // AMDGCN: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]] to i32 () addrspace(5)*
  // AMDGCN: %[[blk_gen_ptr:.*]] = addrspacecast i32 () addrspace(5)* %[[blk_ptr]] to i32 ()*
  // AMDGCN: store i32 ()* %[[blk_gen_ptr]], i32 ()* addrspace(5)* %[[block_B:.*]],
  // AMDGCN: %[[blk_gen_ptr:.*]] = load i32 ()*, i32 ()* addrspace(5)* %[[block_B]]
  // AMDGCN: %[[block_literal:.*]] = bitcast i32 ()* %[[blk_gen_ptr]] to %struct.__opencl_block_literal_generic*
  // AMDGCN: %[[invoke_addr:.*]] = getelementptr inbounds %struct.__opencl_block_literal_generic, %struct.__opencl_block_literal_generic* %[[block_literal]], i32 0, i32 2
  // AMDGCN: %[[blk_gen_ptr:.*]] = bitcast %struct.__opencl_block_literal_generic* %[[block_literal]] to i8*
  // AMDGCN: %[[invoke_func_ptr:.*]] = load i8*, i8** %[[invoke_addr]]
  // AMDGCN: %[[invoke_func:.*]] = bitcast i8* %[[invoke_func_ptr]] to i32 (i8*)*
  // AMDGCN: call {{.*}}i32 %[[invoke_func]](i8* %[[blk_gen_ptr]])

  int (^ block_B)(void) = ^{
    return i;
  };
  block_B();
}

// SPIR-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* %.block_descriptor)
// SPIR:  %[[block:.*]] = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)*
// SPIR:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)* %[[block]], i32 0, i32 3
// SPIR:  %[[block_capture:.*]] = load i32, i32 addrspace(4)* %[[block_capture_addr]]
// AMDGCN-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8* %.block_descriptor)
// AMDGCN:  %[[block:.*]] = bitcast i8* %.block_descriptor to <{ i32, i32, i8*, i32 }>*
// AMDGCN:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }>* %[[block]], i32 0, i32 3
// AMDGCN:  %[[block_capture:.*]] = load i32, i32* %[[block_capture_addr]]

// COMMON-NOT: define{{.*}}@__foo_block_invoke_kernel

// Test that we support block arguments.
// COMMON-LABEL: define {{.*}} @blockArgFunc
int blockArgFunc(int (^ bl)(void)) {
  return bl();
}

// COMMON-LABEL: define {{.*}} @get21
// COMMON: define {{.*}} @__get21_block_invoke
// COMMON: ret i32 21
int get21() {
  return blockArgFunc(^{return 21;});
}

// COMMON-LABEL: define {{.*}} @get42
// COMMON: define {{.*}} @__get42_block_invoke
// COMMON: ret i32 42
int get42() {
  return blockArgFunc(^{return 42;});
}

// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__size"
// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__align"

// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__isa"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__flags"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__reserved"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr"
