// RUN: %clang -foffload-lto=thin -nogpulib -nogpuinc %s -### 2>&1 | FileCheck %s

// CHECK: "--device-compiler=amdgcn-amd-amdhsa=-flto=thin"
// CHECK-SAME: "--device-linker=amdgcn-amd-amdhsa=-plugin-opt=-force-import-all"
// CHECK-SAME: "--device-linker=amdgcn-amd-amdhsa=-plugin-opt=-avail-extern-to-local"
// CHECK-SAME: "--device-linker=amdgcn-amd-amdhsa=-plugin-opt=-avail-extern-gv-in-addrspace-to-local=3"
int main(int, char *[]) {
  return 0;
}
