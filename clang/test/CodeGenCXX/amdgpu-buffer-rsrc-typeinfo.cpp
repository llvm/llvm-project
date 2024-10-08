// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn %s -emit-llvm -o - | FileCheck %s

namespace std { class type_info; };

auto &b = typeid(__amdgpu_buffer_rsrc_t);

// CHECK-DAG: @_ZTSu22__amdgpu_buffer_rsrc_t = {{.*}} c"u22__amdgpu_buffer_rsrc_t\00"
// CHECK-DAG: @_ZTIu22__amdgpu_buffer_rsrc_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu22__amdgpu_buffer_rsrc_t
