// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv64 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s -check-prefixes=CHECK64
// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv64 -emit-llvm %s -x cl -o - | FileCheck %s -check-prefixes=CHECK64
// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv32 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s -check-prefixes=CHECK32
// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv32 -emit-llvm %s -x cl -o - | FileCheck %s -check-prefixes=CHECK32
// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple nvptx64 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s -check-prefixes=NV


// CHECK64: call i64 @llvm.spv.num.workgroups.i64(i32 0)
// CHECK64: call i64 @llvm.spv.num.workgroups.i64(i32 1)
// CHECK64: call i64 @llvm.spv.num.workgroups.i64(i32 2)
// CHECK64: call i64 @llvm.spv.workgroup.size.i64(i32 0)
// CHECK64: call i64 @llvm.spv.workgroup.size.i64(i32 1)
// CHECK64: call i64 @llvm.spv.workgroup.size.i64(i32 2)
// CHECK64: call i64 @llvm.spv.group.id.i64(i32 0)
// CHECK64: call i64 @llvm.spv.group.id.i64(i32 1)
// CHECK64: call i64 @llvm.spv.group.id.i64(i32 2)
// CHECK64: call i64 @llvm.spv.thread.id.in.group.i64(i32 0)
// CHECK64: call i64 @llvm.spv.thread.id.in.group.i64(i32 1)
// CHECK64: call i64 @llvm.spv.thread.id.in.group.i64(i32 2)
// CHECK64: call i64 @llvm.spv.thread.id.i64(i32 0)
// CHECK64: call i64 @llvm.spv.thread.id.i64(i32 1)
// CHECK64: call i64 @llvm.spv.thread.id.i64(i32 2)
// CHECK64: call i64 @llvm.spv.global.size.i64(i32 0)
// CHECK64: call i64 @llvm.spv.global.size.i64(i32 1)
// CHECK64: call i64 @llvm.spv.global.size.i64(i32 2)
// CHECK64: call i64 @llvm.spv.global.offset.i64(i32 0)
// CHECK64: call i64 @llvm.spv.global.offset.i64(i32 1)
// CHECK64: call i64 @llvm.spv.global.offset.i64(i32 2)
// CHECK32: call i32 @llvm.spv.num.workgroups.i32(i32 0)
// CHECK32: call i32 @llvm.spv.num.workgroups.i32(i32 1)
// CHECK32: call i32 @llvm.spv.num.workgroups.i32(i32 2)
// CHECK32: call i32 @llvm.spv.workgroup.size.i32(i32 0)
// CHECK32: call i32 @llvm.spv.workgroup.size.i32(i32 1)
// CHECK32: call i32 @llvm.spv.workgroup.size.i32(i32 2)
// CHECK32: call i32 @llvm.spv.group.id.i32(i32 0)
// CHECK32: call i32 @llvm.spv.group.id.i32(i32 1)
// CHECK32: call i32 @llvm.spv.group.id.i32(i32 2)
// CHECK32: call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
// CHECK32: call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
// CHECK32: call i32 @llvm.spv.thread.id.in.group.i32(i32 2)
// CHECK32: call i32 @llvm.spv.thread.id.i32(i32 0)
// CHECK32: call i32 @llvm.spv.thread.id.i32(i32 1)
// CHECK32: call i32 @llvm.spv.thread.id.i32(i32 2)
// CHECK32: call i32 @llvm.spv.global.size.i32(i32 0)
// CHECK32: call i32 @llvm.spv.global.size.i32(i32 1)
// CHECK32: call i32 @llvm.spv.global.size.i32(i32 2)
// CHECK32: call i32 @llvm.spv.global.offset.i32(i32 0)
// CHECK32: call i32 @llvm.spv.global.offset.i32(i32 1)
// CHECK32: call i32 @llvm.spv.global.offset.i32(i32 2)
// CHECK: call i32 @llvm.spv.subgroup.size()
// CHECK: call i32 @llvm.spv.subgroup.max.size()
// CHECK: call i32 @llvm.spv.num.subgroups()
// CHECK: call i32 @llvm.spv.subgroup.id()
// CHECK: call i32 @llvm.spv.subgroup.local.invocation.id()
  
// NV: call noundef i64 @_Z21__spirv_NumWorkgroupsi(i32 noundef 0) #2
// NV: call noundef i64 @_Z21__spirv_NumWorkgroupsi(i32 noundef 1) #2
// NV: call noundef i64 @_Z21__spirv_NumWorkgroupsi(i32 noundef 2) #2
// NV: call noundef i64 @_Z21__spirv_WorkgroupSizei(i32 noundef 0) #2
// NV: call noundef i64 @_Z21__spirv_WorkgroupSizei(i32 noundef 1) #2
// NV: call noundef i64 @_Z21__spirv_WorkgroupSizei(i32 noundef 2) #2
// NV: call noundef i64 @_Z19__spirv_WorkgroupIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z19__spirv_WorkgroupIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z19__spirv_WorkgroupIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z25__spirv_LocalInvocationIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z25__spirv_LocalInvocationIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z25__spirv_LocalInvocationIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z26__spirv_GlobalInvocationIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z26__spirv_GlobalInvocationIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z26__spirv_GlobalInvocationIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z18__spirv_GlobalSizei(i32 noundef 0) #2
// NV: call noundef i64 @_Z18__spirv_GlobalSizei(i32 noundef 1) #2
// NV: call noundef i64 @_Z18__spirv_GlobalSizei(i32 noundef 2) #2
// NV: call noundef i64 @_Z20__spirv_GlobalOffseti(i32 noundef 0) #2
// NV: call noundef i64 @_Z20__spirv_GlobalOffseti(i32 noundef 1) #2
// NV: call noundef i64 @_Z20__spirv_GlobalOffseti(i32 noundef 2) #2
// NV: call noundef i32 @_Z20__spirv_SubgroupSizev() #2
// NV: call noundef i32 @_Z23__spirv_SubgroupMaxSizev() #2
// NV: call noundef i32 @_Z20__spirv_NumSubgroupsv() #2
// NV: call noundef i32 @_Z18__spirv_SubgroupIdv() #2
// NV: call noundef i32 @_Z33__spirv_SubgroupLocalInvocationIdv() #2

void test_id_and_range() {
  __spirv_NumWorkgroups(0);
  __spirv_NumWorkgroups(1);
  __spirv_NumWorkgroups(2);
  __spirv_WorkgroupSize(0);
  __spirv_WorkgroupSize(1);
  __spirv_WorkgroupSize(2);
  __spirv_WorkgroupId(0);
  __spirv_WorkgroupId(1);
  __spirv_WorkgroupId(2);
  __spirv_LocalInvocationId(0);
  __spirv_LocalInvocationId(1);
  __spirv_LocalInvocationId(2);
  __spirv_GlobalInvocationId(0);
  __spirv_GlobalInvocationId(1);
  __spirv_GlobalInvocationId(2);
  __spirv_GlobalSize(0);
  __spirv_GlobalSize(1);
  __spirv_GlobalSize(2);
  __spirv_GlobalOffset(0);
  __spirv_GlobalOffset(1);
  __spirv_GlobalOffset(2);
  unsigned int ssize = __spirv_SubgroupSize();
  unsigned int smax = __spirv_SubgroupMaxSize();
  unsigned int snum = __spirv_NumSubgroups();
  unsigned int sid = __spirv_SubgroupId();
  unsigned int sinvocid = __spirv_SubgroupLocalInvocationId();
}
