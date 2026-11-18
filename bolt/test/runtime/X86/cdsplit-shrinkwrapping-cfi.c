// Check that shrink wrapping initializes the CFA at a warm-fragment boundary.

// REQUIRES: system-linux, bolt-runtime, target=x86_64-{{.*}}
// RUN: %clang %cflags -O3 -fno-inline %s -Wl,-q -o %t.exe
// RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
// RUN:   -o %t.instrumented
// RUN: %t.instrumented
// RUN: llvm-bolt %t.exe --data=%t.fdata --reorder-blocks=ext-tsp \
// RUN:   --split-functions --split-strategy=cdsplit --frame-opt=hot \
// RUN:   -o %t.bolt | FileCheck %s --check-prefix=SHRINK
// RUN: llvm-nm --format=posix --radix=x %t.bolt > %t.frames
// RUN: llvm-dwarfdump --eh-frame %t.bolt >> %t.frames
// RUN: FileCheck %s < %t.frames

// SHRINK: Shrink wrapping moved 1 spills inserting load/stores
// CHECK: dispatch.warm t [[WARM:[0-9a-f]+]]
// CHECK: FDE {{.*}} pc={{0*}}[[WARM]]...
// CHECK-NEXT: Format: DWARF32
// CHECK-NEXT: DW_CFA_def_cfa_offset: +32

volatile unsigned gate = 1;

long identity(long value) { return value; }

long (*callback)(long) = identity;

long dispatch(long value, unsigned warm) {
  volatile long local = 0;
  if (warm) {
    if (!gate)
      return 0;
    long saved = gate;
    value = callback(value) + saved;
  }
  return value + local;
}

int main(void) {
  for (unsigned i = 0; i < 100; ++i)
    dispatch(i, i == 99);
}
