// RUN: %clang -g -gheterogeneous-dwarf -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NO_DEBUG
int main (void) {
  return 0;
}

// CHECK:  i32 2, !"Debug Info Version", i32 4}
// NO_DEBUG-NOT: !"Debug Info Version"
