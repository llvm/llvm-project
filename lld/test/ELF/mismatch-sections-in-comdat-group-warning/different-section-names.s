// RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/func.s -o %t0.o
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
// RUN: ld.lld -shared %t0.o %t1.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s
// RUN: ld.lld -shared %t1.o %t0.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s --check-prefix=REVERSE

// CHECK:   warning: comdat group with signature 'func' in '{{.*}}0.o' does not have section '.data.func2' which is part of comdat group in '{{.*}}1.o'
// REVERSE: warning: comdat group with signature 'func' in '{{.*}}1.o' does not have section '.data.func' which is part of comdat group in '{{.*}}0.o'

  .global func
  .section .data.func2,"awG",@progbits,func,comdat
func:
  .word 7
