// Check that debug locations are updated when BOLT rewrites ADRP+ADD pairs.

// RUN: %clang %cflags -O2 -g -fuse-ld=lld -Wl,-q %s -o %t
// RUN: llvm-bolt --update-debug-sections %t -o %t.bolt

// RUN: llvm-dwarfdump --debug-line %t.bolt > %t.dump
// RUN: llvm-objdump --disassemble=table_addr %t.bolt 2>&1 >> %t.dump
// RUN: FileCheck %s < %t.dump

// CHECK: [[#%x,ADDR:]] {{.*}} is_stmt prologue_end
// CHECK: [[#ADDR]] <table_addr>:
// CHECK: [[#ADDR]]: {{.*}} adrp

int table[1] = {1};

__attribute__((noinline)) int *table_addr() { return table; }

int main(void) { return table_addr()[0]; }
