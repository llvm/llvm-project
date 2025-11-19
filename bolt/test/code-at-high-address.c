// Check that llvm-bolt pushes code to higher addresses under
// --hot-functions-at-end when rewriting code in-place.

// REQUIRES: system-linux

// RUN: %clang %cflags -O0 %s -o %t -no-pie -Wl,-q -falign-functions=64 \
// RUN:   -nostartfiles -nostdlib -ffreestanding
// RUN: llvm-bolt %t -o %t.bolt --use-old-text --align-functions=1 \
// RUN:   --no-huge-pages --align-text=1 --use-gnu-stack --hot-functions-at-end \
// RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
// RUN: llvm-readelf --sections %t.bolt | FileCheck %s

// CHECK-BOLT: using original .text for new code with 0x1 alignment at {{.*}}

// As .text is pushed higher, preceding .bolt.org.text should have non-zero
// size.
// CHECK: .bolt.org.text PROGBITS
// CHECK-NOT: {{ 000000 }}
// CHECK-SAME: AX
// CHECK-NEXT: .text PROGBITS

int foo() { return 0; }

int main() { return foo(); }
