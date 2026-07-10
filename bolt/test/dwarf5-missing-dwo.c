// Check that llvm-bolt correctly reports a missing DWO file while updating
// debug info.
//
// REQUIRES: system-linux
//
// RUN: %clang %cflags -g -dwarf5 -gsplit-dwarf=single -c %s -o %t.o
// RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
// RUN: rm %t.o
// RUN: not llvm-bolt %t.exe -o %t.bolt --update-debug-sections \
// RUN:   2>&1 | FileCheck %s -DDWO=%t.o
//
// Check that llvm-bolt succeeds on the same binary with disabled debug info
// update.
//
// RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections=0

// CHECK:      BOLT-ERROR: unable to load [[DWO]]
// CHECK-NEXT: BOLT-ERROR: 1 required DWO file(s) not found

int main() { return 0; }
