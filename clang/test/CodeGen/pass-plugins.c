// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE
// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -mllvm -wave-goodbye 2>&1 | FileCheck %s --check-prefix=CHECK-ACTIVE
// REQUIRES: plugins, llvm-examples
// UNSUPPORTED: target={{.*windows.*}}
// CHECK-INACTIVE-NOT: Bye
// CHECK-ACTIVE: Bye: f

int f(int x) {
  return x;
}
