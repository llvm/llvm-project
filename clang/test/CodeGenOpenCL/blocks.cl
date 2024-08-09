// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=COMMON,AMDGCN %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple spir-unknown-unknown | FileCheck -check-prefixes=CHECK-DEBUG %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=CHECK-DEBUG %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_device_enqueue,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_device_enqueue,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables  -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=COMMON,AMDGCN %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_device_enqueue,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables  -emit-llvm -o - -O0 -debug-info-kind=limited -triple spir-unknown-unknown | FileCheck -check-prefixes=CHECK-DEBUG %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_device_enqueue,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -emit-llvm -o - -O0 -debug-info-kind=limited -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=CHECK-DEBUG %s

// SPIR: @__block_literal_global = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 12, i32 4, ptr addrspace(4) addrspacecast (ptr @block_A_block_invoke to ptr addrspace(4)) }
// AMDGCN: @__block_literal_global = internal addrspace(1) constant { i32, i32, ptr } { i32 16, i32 8, ptr @block_A_block_invoke }
// COMMON-NOT: .str

// SPIR-LABEL: define internal {{.*}}void @block_A_block_invoke(ptr addrspace(4) noundef %.block_descriptor, ptr addrspace(3) noundef %a)
// AMDGCN-LABEL: define internal {{.*}}void @block_A_block_invoke(ptr noundef %.block_descriptor, ptr addrspace(3) noundef %a)
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
  // SPIR: %[[block_size:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr addrspace(4), i32 }>, ptr %block, i32 0, i32 0
  // AMDGCN: %[[block_size:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr, i32 }>, ptr addrspace(5) %block, i32 0, i32 0
  // SPIR: store i32 16, ptr %[[block_size]]
  // AMDGCN: store i32 20, ptr addrspace(5) %[[block_size]]
  // SPIR: %[[block_align:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr addrspace(4), i32 }>, ptr %block, i32 0, i32 1
  // AMDGCN: %[[block_align:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr, i32 }>, ptr addrspace(5) %block, i32 0, i32 1
  // SPIR: store i32 4, ptr %[[block_align]]
  // AMDGCN: store i32 8, ptr addrspace(5) %[[block_align]]
  // SPIR: %[[block_invoke:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr addrspace(4), i32 }>, ptr %[[block:.*]], i32 0, i32 2
  // SPIR: store ptr addrspace(4) addrspacecast (ptr @__foo_block_invoke to ptr addrspace(4)), ptr %[[block_invoke]]
  // SPIR: %[[block_captured:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr addrspace(4), i32 }>, ptr %[[block]], i32 0, i32 3
  // SPIR: %[[i_value:.*]] = load i32, ptr %i
  // SPIR: store i32 %[[i_value]], ptr %[[block_captured]],
  // SPIR: %[[blk_gen_ptr:.*]] = addrspacecast ptr %[[block]] to ptr addrspace(4)
  // SPIR: store ptr addrspace(4) %[[blk_gen_ptr]], ptr %[[block_B:.*]],
  // SPIR: %[[block_literal:.*]] = load ptr addrspace(4), ptr %[[block_B]]
  // SPIR: call {{.*}}i32 @__foo_block_invoke(ptr addrspace(4) noundef %[[block_literal]])
  // AMDGCN: %[[block_invoke:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr, i32 }>, ptr addrspace(5) %[[block:.*]], i32 0, i32 2
  // AMDGCN: store ptr @__foo_block_invoke, ptr addrspace(5) %[[block_invoke]]
  // AMDGCN: %[[block_captured:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr, i32 }>, ptr addrspace(5) %[[block]], i32 0, i32 3
  // AMDGCN: %[[i_value:.*]] = load i32, ptr addrspace(5) %i
  // AMDGCN: store i32 %[[i_value]], ptr addrspace(5) %[[block_captured]],
  // AMDGCN: %[[blk_gen_ptr:.*]] = addrspacecast ptr addrspace(5) %[[block]] to ptr
  // AMDGCN: store ptr %[[blk_gen_ptr]], ptr addrspace(5) %[[block_B:.*]],
  // AMDGCN: %[[block_literal:.*]] = load ptr, ptr addrspace(5) %[[block_B]]
  // AMDGCN: call {{.*}}i32 @__foo_block_invoke(ptr noundef %[[block_literal]])

  int (^ block_B)(void) = ^{
    return i;
  };
  block_B();
}

// SPIR-LABEL: define internal {{.*}}i32 @__foo_block_invoke(ptr addrspace(4) noundef %.block_descriptor)
// SPIR:  %[[block_capture_addr:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr addrspace(4), i32 }>, ptr addrspace(4) %.block_descriptor, i32 0, i32 3
// SPIR:  %[[block_capture:.*]] = load i32, ptr addrspace(4) %[[block_capture_addr]]
// AMDGCN-LABEL: define internal {{.*}}i32 @__foo_block_invoke(ptr noundef %.block_descriptor)
// AMDGCN:  %[[block_capture_addr:.*]] = getelementptr inbounds nuw <{ i32, i32, ptr, i32 }>, ptr %.block_descriptor, i32 0, i32 3
// AMDGCN:  %[[block_capture:.*]] = load i32, ptr %[[block_capture_addr]]

// COMMON-NOT: define{{.*}}@__foo_block_invoke_kernel

// COMMON-LABEL: define {{.*}}@call_block
// call {{.*}}@__call_block_block_invoke
int call_block() {
  return ^int(int num) { return num; } (11);
}

// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__size"
// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__align"

// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__isa"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__flags"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__reserved"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr"
