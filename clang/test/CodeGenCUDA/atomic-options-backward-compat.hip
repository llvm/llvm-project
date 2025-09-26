// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1100 -emit-llvm -o - \
// RUN:   -fno-atomic-backward-compatible %s | FileCheck %s --check-prefix=CHECK-NEW
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1100 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-OLD

// CHECK-NEW-LABEL: @_Z16test_atomic_faddPf(
// CHECK-NEW: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}
// CHECK-OLD-LABEL: @_Z16test_atomic_faddPf(
// CHECK-OLD: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4
void test_atomic_fadd(float* ptr) {
  __atomic_fetch_add(ptr, 1.0f, __ATOMIC_RELAXED);
}

// CHECK-NEW-LABEL: @_Z13test_overridePf(
// CHECK-NEW: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+}}
// CHECK-OLD-LABEL: @_Z13test_overridePf(
// CHECK-OLD-NOT: !amdgpu.no.fine.grained.memory
// CHECK-OLD-NOT: !amdgpu.no.remote.memory
// CHECK-OLD: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.ignore.denormal.mode !{{[0-9]+}}
void test_override(float* ptr) {
  [[clang::atomic(ignore_denormal_mode)]] {
    __atomic_fetch_add(ptr, 1.0f, __ATOMIC_RELAXED);
  }
}

// CHECK-NEW-LABEL: @_Z20test_nested_overridePf(
// First atomic in the outer scope should have ignore_denormal plus the two new metadata.
// CHECK-NEW: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+}}
// For the inner scope with no_ignore_denormal_mode, ensure ignore.denormal is absent but the two new metadata are present.
// CHECK-NEW: atomicrmw fsub ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}
// CHECK-NEW-NOT: !amdgpu.ignore.denormal.mode
// CHECK-OLD-LABEL: @_Z20test_nested_overridePf(
// CHECK-OLD-NOT: !amdgpu.no.fine.grained.memory
// CHECK-OLD-NOT: !amdgpu.no.remote.memory
// CHECK-OLD: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.ignore.denormal.mode !{{[0-9]+}}
void test_nested_override(float* ptr) {
  [[clang::atomic(ignore_denormal_mode)]] {
    __atomic_fetch_add(ptr, 1.0f, __ATOMIC_RELAXED);

    // CHECK-OLD-NOT: !amdgpu.no.fine.grained.memory
    // CHECK-OLD-NOT: !amdgpu.no.remote.memory
    // CHECK-OLD-NOT: !amdgpu.ignore.denormal.mode
    // CHECK-OLD: atomicrmw fsub ptr %{{.*}}, float %{{.*}} monotonic, align 4
    [[clang::atomic(no_ignore_denormal_mode)]] {
      __atomic_fetch_sub(ptr, 1.0f, __ATOMIC_RELAXED);
    }
  }
}

// CHECK-NEW-LABEL: @_Z26test_fine_grained_overridePf(
// CHECK-NEW: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.remote.memory !{{[0-9]+}}
// CHECK-OLD-LABEL: @_Z26test_fine_grained_overridePf(
// CHECK-OLD: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4
// CHECK-OLD-NOT: !amdgpu.no.fine.grained.memory
// CHECK-OLD-NOT: !amdgpu.no.remote.memory
void test_fine_grained_override(float* ptr) {
  [[clang::atomic(fine_grained_memory)]] {
    __atomic_fetch_add(ptr, 1.0f, __ATOMIC_RELAXED);
  }
}

// CHECK-NEW-LABEL: @_Z29test_no_fine_grained_overridePf(
// CHECK-NEW: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}
// CHECK-OLD-LABEL: @_Z29test_no_fine_grained_overridePf(
// CHECK-OLD: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory
// CHECK-OLD-NOT: !amdgpu.no.remote.memory
void test_no_fine_grained_override(float* ptr) {
  [[clang::atomic(no_fine_grained_memory)]] {
    __atomic_fetch_add(ptr, 1.0f, __ATOMIC_RELAXED);
  }
}
