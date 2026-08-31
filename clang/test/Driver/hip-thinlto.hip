// RUN: %clang -foffload-lto=thin -nogpulib -nogpuinc %s -### 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: "-flto=thin"
// CHECK-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// CHECK: clang-linker-wrapper
// CHECK-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
// CHECK-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// CHECK-NOT: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-force-import-all"
int main(int, char *[]) {
  return 0;
}
