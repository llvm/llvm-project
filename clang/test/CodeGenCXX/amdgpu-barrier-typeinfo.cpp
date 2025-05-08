// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn %s -emit-llvm -o - | FileCheck %s

namespace std { class type_info; };

auto &b0 = typeid(__amdgpu_named_workgroup_barrier_t);
auto &b1 = typeid(__amdgpu_named_cluster_barrier_t);

// CHECK-DAG: @_ZTSu34__amdgpu_named_workgroup_barrier_t = {{.*}} c"u34__amdgpu_named_workgroup_barrier_t\00"
// CHECK-DAG: @_ZTIu34__amdgpu_named_workgroup_barrier_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu34__amdgpu_named_workgroup_barrier_t
// CHECK-DAG: @_ZTSu32__amdgpu_named_cluster_barrier_t = {{.*}} c"u32__amdgpu_named_cluster_barrier_t\00"
// CHECK-DAG: @_ZTIu32__amdgpu_named_cluster_barrier_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu32__amdgpu_named_cluster_barrier_t

