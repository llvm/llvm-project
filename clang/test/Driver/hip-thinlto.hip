// RUN: %clang -foffload-lto=thin -nogpulib -nogpuinc %s -### 2>&1 | FileCheck %s

// CHECK: -plugin-opt=thinlto
// CHECK-SAME: -plugin-opt=-force-import-all
// CHECK-SAME: -plugin-opt=-avail-extern-to-local
int main(int, char *[]) {
  return 0;
}
