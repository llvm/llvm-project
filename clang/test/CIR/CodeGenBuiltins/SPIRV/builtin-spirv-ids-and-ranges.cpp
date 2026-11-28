// RUN: %clang_cc1 -triple spirv64 -fsycl-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR64 --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64 -fsycl-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM64 --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple spirv64 -fsycl-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM64 --input-file=%t.ll %s

// RUN: %clang_cc1 -triple spirv32 -fsycl-is-device -fclangir -emit-cir %s -o %t32.cir
// RUN: FileCheck --check-prefixes=CIR,CIR32 --input-file=%t32.cir %s
// RUN: %clang_cc1 -triple spirv32 -fsycl-is-device -fclangir -emit-llvm %s -o %t32-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM32 --input-file=%t32-cir.ll %s
// RUN: %clang_cc1 -triple spirv32 -fsycl-is-device -emit-llvm %s -o %t32.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM32 --input-file=%t32.ll %s

// Tests CIR emission of the SPIR-V work-item and subgroup query builtins.
// Mirrors clang/test/CodeGenSPIRV/Builtins/ids_and_ranges.c.

// CIR-LABEL: cir.func {{.*}} @_Z19test_num_workgroupsv
// CIR64: cir.call_llvm_intrinsic "spv.num.workgroups" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.num.workgroups" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z19test_num_workgroupsv(
// LLVM64: call i64 @llvm.spv.num.workgroups.i64(i32 1)
// LLVM32: call i32 @llvm.spv.num.workgroups.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_num_workgroups() {
  return __builtin_spirv_num_workgroups(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z19test_workgroup_sizev
// CIR64: cir.call_llvm_intrinsic "spv.workgroup.size" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.workgroup.size" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z19test_workgroup_sizev(
// LLVM64: call i64 @llvm.spv.workgroup.size.i64(i32 1)
// LLVM32: call i32 @llvm.spv.workgroup.size.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_workgroup_size() {
  return __builtin_spirv_workgroup_size(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z17test_workgroup_idv
// CIR64: cir.call_llvm_intrinsic "spv.group.id" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.group.id" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z17test_workgroup_idv(
// LLVM64: call i64 @llvm.spv.group.id.i64(i32 1)
// LLVM32: call i32 @llvm.spv.group.id.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_workgroup_id() {
  return __builtin_spirv_workgroup_id(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z24test_local_invocation_idv
// CIR64: cir.call_llvm_intrinsic "spv.thread.id.in.group" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.thread.id.in.group" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z24test_local_invocation_idv(
// LLVM64: call i64 @llvm.spv.thread.id.in.group.i64(i32 1)
// LLVM32: call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_local_invocation_id() {
  return __builtin_spirv_local_invocation_id(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z25test_global_invocation_idv
// CIR64: cir.call_llvm_intrinsic "spv.thread.id" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.thread.id" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z25test_global_invocation_idv(
// LLVM64: call i64 @llvm.spv.thread.id.i64(i32 1)
// LLVM32: call i32 @llvm.spv.thread.id.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_global_invocation_id() {
  return __builtin_spirv_global_invocation_id(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z16test_global_sizev
// CIR64: cir.call_llvm_intrinsic "spv.global.size" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.global.size" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z16test_global_sizev(
// LLVM64: call i64 @llvm.spv.global.size.i64(i32 1)
// LLVM32: call i32 @llvm.spv.global.size.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_global_size() {
  return __builtin_spirv_global_size(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z18test_global_offsetv
// CIR64: cir.call_llvm_intrinsic "spv.global.offset" %{{.*}} : (!s32i) -> !u64i
// CIR32: cir.call_llvm_intrinsic "spv.global.offset" %{{.*}} : (!s32i) -> !u32i
// LLVM-LABEL: define{{.*}} @_Z18test_global_offsetv(
// LLVM64: call i64 @llvm.spv.global.offset.i64(i32 1)
// LLVM32: call i32 @llvm.spv.global.offset.i32(i32 1)
[[clang::sycl_external]] __SIZE_TYPE__ test_global_offset() {
  return __builtin_spirv_global_offset(1);
}

// CIR-LABEL: cir.func {{.*}} @_Z18test_subgroup_sizev
// CIR: cir.call_llvm_intrinsic "spv.subgroup.size" : () -> !u32i
// LLVM-LABEL: define{{.*}} @_Z18test_subgroup_sizev(
// LLVM: call i32 @llvm.spv.subgroup.size()
[[clang::sycl_external]] unsigned int test_subgroup_size() {
  return __builtin_spirv_subgroup_size();
}

// CIR-LABEL: cir.func {{.*}} @_Z22test_subgroup_max_sizev
// CIR: cir.call_llvm_intrinsic "spv.subgroup.max.size" : () -> !u32i
// LLVM-LABEL: define{{.*}} @_Z22test_subgroup_max_sizev(
// LLVM: call i32 @llvm.spv.subgroup.max.size()
[[clang::sycl_external]] unsigned int test_subgroup_max_size() {
  return __builtin_spirv_subgroup_max_size();
}

// CIR-LABEL: cir.func {{.*}} @_Z18test_num_subgroupsv
// CIR: cir.call_llvm_intrinsic "spv.num.subgroups" : () -> !u32i
// LLVM-LABEL: define{{.*}} @_Z18test_num_subgroupsv(
// LLVM: call i32 @llvm.spv.num.subgroups()
[[clang::sycl_external]] unsigned int test_num_subgroups() {
  return __builtin_spirv_num_subgroups();
}

// CIR-LABEL: cir.func {{.*}} @_Z16test_subgroup_idv
// CIR: cir.call_llvm_intrinsic "spv.subgroup.id" : () -> !u32i
// LLVM-LABEL: define{{.*}} @_Z16test_subgroup_idv(
// LLVM: call i32 @llvm.spv.subgroup.id()
[[clang::sycl_external]] unsigned int test_subgroup_id() {
  return __builtin_spirv_subgroup_id();
}

// CIR-LABEL: cir.func {{.*}} @_Z33test_subgroup_local_invocation_idv
// CIR: cir.call_llvm_intrinsic "spv.subgroup.local.invocation.id" : () -> !u32i
// LLVM-LABEL: define{{.*}} @_Z33test_subgroup_local_invocation_idv(
// LLVM: call i32 @llvm.spv.subgroup.local.invocation.id()
[[clang::sycl_external]] unsigned int test_subgroup_local_invocation_id() {
  return __builtin_spirv_subgroup_local_invocation_id();
}
