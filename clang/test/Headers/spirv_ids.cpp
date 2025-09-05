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
  
// NV: call noundef i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 noundef 0) #2
// NV: call noundef i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 noundef 1) #2
// NV: call noundef i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 noundef 2) #2
// NV: call noundef i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 noundef 0) #2
// NV: call noundef i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 noundef 1) #2
// NV: call noundef i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 noundef 2) #2
// NV: call noundef i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 noundef 0) #2
// NV: call noundef i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 noundef 1) #2
// NV: call noundef i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 noundef 2) #2
// NV: call noundef i64 @_Z25__spirv_BuiltInGlobalSizei(i32 noundef 0) #2
// NV: call noundef i64 @_Z25__spirv_BuiltInGlobalSizei(i32 noundef 1) #2
// NV: call noundef i64 @_Z25__spirv_BuiltInGlobalSizei(i32 noundef 2) #2
// NV: call noundef i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 noundef 0) #2
// NV: call noundef i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 noundef 1) #2
// NV: call noundef i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 noundef 2) #2
// NV: call noundef i32 @_Z27__spirv_BuiltInSubgroupSizev() #2
// NV: call noundef i32 @_Z30__spirv_BuiltInSubgroupMaxSizev() #2
// NV: call noundef i32 @_Z27__spirv_BuiltInNumSubgroupsv() #2
// NV: call noundef i32 @_Z25__spirv_BuiltInSubgroupIdv() #2
// NV: call noundef i32 @_Z40__spirv_BuiltInSubgroupLocalInvocationIdv() #2

[[clang::sycl_external]] void test_id_and_range() {
  __spirv_BuiltInNumWorkgroups(0);
  __spirv_BuiltInNumWorkgroups(1);
  __spirv_BuiltInNumWorkgroups(2);
  __spirv_BuiltInWorkgroupSize(0);
  __spirv_BuiltInWorkgroupSize(1);
  __spirv_BuiltInWorkgroupSize(2);
  __spirv_BuiltInWorkgroupId(0);
  __spirv_BuiltInWorkgroupId(1);
  __spirv_BuiltInWorkgroupId(2);
  __spirv_BuiltInLocalInvocationId(0);
  __spirv_BuiltInLocalInvocationId(1);
  __spirv_BuiltInLocalInvocationId(2);
  __spirv_BuiltInGlobalInvocationId(0);
  __spirv_BuiltInGlobalInvocationId(1);
  __spirv_BuiltInGlobalInvocationId(2);
  __spirv_BuiltInGlobalSize(0);
  __spirv_BuiltInGlobalSize(1);
  __spirv_BuiltInGlobalSize(2);
  __spirv_BuiltInGlobalOffset(0);
  __spirv_BuiltInGlobalOffset(1);
  __spirv_BuiltInGlobalOffset(2);
  unsigned int ssize = __spirv_BuiltInSubgroupSize();
  unsigned int smax = __spirv_BuiltInSubgroupMaxSize();
  unsigned int snum = __spirv_BuiltInNumSubgroups();
  unsigned int sid = __spirv_BuiltInSubgroupId();
  unsigned int sinvocid = __spirv_BuiltInSubgroupLocalInvocationId();
}
