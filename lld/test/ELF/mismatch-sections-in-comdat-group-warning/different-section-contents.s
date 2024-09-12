// RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/func.s -o %t0.o
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
// RUN: ld.lld -shared %t0.o %t1.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s
// RUN: ld.lld -shared %t1.o %t0.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s --check-prefix=REVERSE

// CHECK:   warning: section '.data.func' for comdat group with signature 'func' has different contents between '{{.*}}1.o' and '{{.*}}0.o'
// REVERSE: warning: section '.data.func' for comdat group with signature 'func' has different contents between '{{.*}}0.o' and '{{.*}}1.o'

  .global func
  .section .data.func,"awG",@progbits,func,comdat
func:
  .word 6
