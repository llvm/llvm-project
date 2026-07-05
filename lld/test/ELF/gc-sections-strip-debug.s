# REQUIRES: x86
## Test that we don't strip SHF_ALLOC .debug* or crash (PR48071
## mark liveness of a merge section which has not been split).

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o --gc-sections --strip-debug -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK: .debug_gdb_scripts

.globl _start
_start:
  leaq .L.str(%rip), %rax

.section .debug_gdb_scripts,"aMS",@progbits,1
.L.str:
  .asciz "Rust uses SHF_ALLOC .debug_gdb_scripts"
