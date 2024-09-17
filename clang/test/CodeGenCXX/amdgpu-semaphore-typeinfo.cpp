// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn %s -emit-llvm -o - | FileCheck %s

namespace std { class type_info; };

auto &b0 = typeid(__amdgpu_semaphore0_t);
auto &b7 = typeid(__amdgpu_semaphore7_t);

// CHECK-DAG: @_ZTSu21__amdgpu_semaphore0_t = {{.*}} c"u21__amdgpu_semaphore0_t\00"
// CHECK-DAG: @_ZTIu21__amdgpu_semaphore0_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu21__amdgpu_semaphore0_t
// CHECK-DAG: @_ZTSu21__amdgpu_semaphore7_t = {{.*}} c"u21__amdgpu_semaphore7_t\00"
// CHECK-DAG: @_ZTIu21__amdgpu_semaphore7_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu21__amdgpu_semaphore7_t
