// This test checks that reorder-data pass puts new hot .data section
// to the writable segment.

// RUN: %clang %cflags -O3 -nostdlib -Wl,-q %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt --reorder-data=".data" \
// RUN:   -data %S/Inputs/reorder-data-writable-ptload.fdata
// RUN: llvm-readelf -SlW %t.bolt | FileCheck %s

// CHECK: .bolt.org.data
// CHECK: {{.*}} .data PROGBITS [[#%x,ADDR:]] [[#%x,OFF:]]
// CHECK: LOAD 0x{{.*}}[[#OFF]] 0x{{.*}}[[#ADDR]] {{.*}} RW

volatile int cold1 = 42;
volatile int hot1 = 42;
volatile int cold2 = 42;
volatile int cold3 = 42;

void _start() {
  hot1++;
  _start();
}
