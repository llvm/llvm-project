// RUN: %clang -target x86_64-linux-gnu -g -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=NODWARF64
// RUN: %clang -target x86_64-linux-gnu -g -gdwarf64 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=DWARF64
// RUN: %clang -target x86_64-linux-gnu -g -gdwarf64 -gdwarf32 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=NODWARF64

// DWARF64: !{i32 7, !"DWARF64", i32 1}
// NODWARF64-NOT: !"DWARF64"

int main (void) {
  return 0;
}
