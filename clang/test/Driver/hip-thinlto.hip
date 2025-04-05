// RUN: %clang -foffload-lto=thin --offload-link %s -### 2>&1 | FileCheck %s

// CHECK: "-plugin-opt=-force-import-all"{{.*}}"-plugin-opt=-avail-extern-to-local"
int main(int, char *[]) {
  return 0;
}
