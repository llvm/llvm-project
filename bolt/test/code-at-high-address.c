// Check that llvm-bolt pushes code to higher addresses under
// --hot-functions-at-end when rewriting code in-place.

// REQUIRES: system-linux

// RUN: %clang %cflags -O0 %s -o %t -no-pie -Wl,-q -falign-functions=64 \
// RUN:   -nostartfiles -nostdlib -ffreestanding
// RUN: link_fdata %s %t %t.fdata
// RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --use-old-text \
// RUN:   --align-functions=1 --no-huge-pages --align-text=1 --use-gnu-stack \
// RUN:   --reorder-functions=cdsort \
// RUN:    --hot-functions-at-end | FileCheck %s --check-prefix=CHECK-BOLT
// RUN: llvm-readelf --sections %t.bolt | FileCheck %s

// CHECK-BOLT: using original .text for new code with 0x1 alignment at {{.*}}

// As .text is pushed higher, preceding .bolt.org.text should have non-zero
// size.
// CHECK: .bolt.org.text PROGBITS
// CHECK-NOT: {{ 000000 }}
// CHECK-SAME: AX
// CHECK-NEXT: .text.cold PROGBITS
// CHECK-NEXT: .text PROGBITS

// FDATA: 0 [unknown] 0 1 foo 0 0 1
int foo() { return 0; }

// Cold function.
int bar() { return 42; }

// FDATA: 0 [unknown] 0 1 main 0 0 1
int main(int argc, char **argv) { return argc ? foo() : bar(); }
