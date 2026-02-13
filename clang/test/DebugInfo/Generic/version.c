// REQUIRES: !system-windows, !system-cygwin
// RUN: %clang -g -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NO_DEBUG
int main (void) {
  return 0;
}

// CHECK:  i32 2, !"Debug Info Version", i32 3}
// NO_DEBUG-NOT: !"Debug Info Version"
