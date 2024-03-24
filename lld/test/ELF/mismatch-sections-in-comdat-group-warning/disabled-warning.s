// RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/func.s -o %t0.o
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
// RUN: ld.lld -shared %t0.o %t1.o -o /dev/null --fatal-warnings
// RUN: ld.lld -shared %t1.o %t0.o -o /dev/null --fatal-warnings
// RUN: ld.lld -shared %t0.o %t1.o -o /dev/null --fatal-warnings --no-warn-mismatch-sections-in-comdat-groups
// RUN: ld.lld -shared %t1.o %t0.o -o /dev/null --fatal-warnings --no-warn-mismatch-sections-in-comdat-groups

// No error since the warning is disabled, despite the mismatch.

  .global func
  .section .data.func,"awG",@progbits,func,comdat
func:
  .word 6
