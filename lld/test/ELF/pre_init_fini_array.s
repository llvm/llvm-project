// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2
// RUN: ld.lld %t2 -o %t2.so -shared -soname=so
// RUN: ld.lld %t %t2.so -o %t2
// RUN: llvm-readelf -S -d -r -s %t2 | FileCheck %s
// RUN: llvm-objdump -d --syms %t2 | FileCheck --check-prefix=DISASM %s

.globl _start
_start:
  call *__preinit_array_start
  call *__preinit_array_end
  call *__init_array_start
  call *__init_array_end
  call *__fini_array_start
  call *__fini_array_end


.section .init_array,"aw",@init_array
  .quad 0

.section .preinit_array,"aw",@preinit_array
        .quad 0
        .byte 0

.section .fini_array,"aw",@fini_array
        .quad 0
        .short 0

// CHECK-LABEL: Section Headers:
// CHECK:      Name           Type          Address                 Off       Size                    ES Flg
// CHECK:      .init_array    INIT_ARRAY    [[# %x, INIT_ADDR:]]    [[# %x,]] [[# %x, INIT_SIZE:]]    00 WA
// CHECK-NEXT: .preinit_array PREINIT_ARRAY [[# %x, PREINIT_ADDR:]] [[# %x,]] [[# %x, PREINIT_SIZE:]] 00 WA
// CHECK-NEXT: .fini_array    FINI_ARRAY    [[# %x, FINI_ADDR:]]    [[# %x,]] [[# %x, FINI_SIZE:]]    00 WA

// CHECK-LABEL: Dynamic section
// CHECK: (PREINIT_ARRAY)        0x[[# PREINIT_ADDR]]
// CHECK: (PREINIT_ARRAYSZ)      [[# %u, PREINIT_SIZE]] (bytes)
// CHECK: (INIT_ARRAY)           0x[[# INIT_ADDR]]
// CHECK: (INIT_ARRAYSZ)         [[# %u, INIT_SIZE]] (bytes)
// CHECK: (FINI_ARRAY)           0x[[# FINI_ADDR]]
// CHECK: (FINI_ARRAYSZ)         [[# %u, FINI_SIZE]] (bytes)

// CHECK-LABEL:      There are no relocations in this file.

// CHECK-LABEL: Symbol table '.symtab'
// CHECK:       Value                             Size Type    Bind   Vis       Ndx   Name
// CHECK:       [[# FINI_ADDR + FINI_SIZE]]       0    NOTYPE  LOCAL  HIDDEN    [[#]] __fini_array_end
// CHECK-NEXT:  [[# FINI_ADDR]]                   0    NOTYPE  LOCAL  HIDDEN    [[#]] __fini_array_start
// CHECK-NEXT:  [[# INIT_ADDR + INIT_SIZE]]       0    NOTYPE  LOCAL  HIDDEN    [[#]] __init_array_end
// CHECK-NEXT:  [[# INIT_ADDR]]                   0    NOTYPE  LOCAL  HIDDEN    [[#]] __init_array_start
// CHECK-NEXT:  [[# PREINIT_ADDR + PREINIT_SIZE]] 0    NOTYPE  LOCAL  HIDDEN    [[#]] __preinit_array_end
// CHECK-NEXT:  [[# PREINIT_ADDR]]                0    NOTYPE  LOCAL  HIDDEN    [[#]] __preinit_array_start

// DISASM:      SYMBOL TABLE:
// DISASM-DAG: {{0*}}[[# %x, PREINIT_ARRAY_START:]]  l  .preinit_array  {{0+}}  .hidden  __preinit_array_start
// DISASM-DAG: {{0*}}[[# %x, PREINIT_ARRAY_END:]]    l  .preinit_array  {{0+}}  .hidden  __preinit_array_end
// DISASM-DAG: {{0*}}[[# %x, INIT_ARRAY_START:]]     l  .init_array  {{0+}}  .hidden  __init_array_start
// DISASM-DAG: {{0*}}[[# %x, INIT_ARRAY_END:]]       l  .init_array  {{0+}}  .hidden  __init_array_end
// DISASM-DAG: {{0*}}[[# %x, FINI_ARRAY_START:]]     l  .fini_array  {{0+}}  .hidden  __fini_array_start
// DISASM-DAG: {{0*}}[[# %x, FINI_ARRAY_END:]]       l  .fini_array  {{0+}}  .hidden  __fini_array_end

// DISASM:      <_start>:
// DISASM-NEXT:   callq   *[[# %u, PREINIT_ARRAY_START]]
// DISASM-NEXT:   callq   *[[# %u, PREINIT_ARRAY_END]]
// DISASM-NEXT:   callq   *[[# %u, INIT_ARRAY_START]]
// DISASM-NEXT:   callq   *[[# %u, INIT_ARRAY_END]]
// DISASM-NEXT:   callq   *[[# %u, FINI_ARRAY_START]]
// DISASM-NEXT:   callq   *[[# %u, FINI_ARRAY_END]]
