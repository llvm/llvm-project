// RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/func.s -o %t0.o
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
// RUN: ld.lld -shared %t0.o %t1.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s
// RUN: ld.lld -shared %t1.o %t0.o -o /dev/null --warn-mismatch-sections-in-comdat-groups 2>&1 | FileCheck %s --check-prefix=REVERSE

// CHECK:   warning: comdat group with signature 'func' in '{{.*}}0.o' has 1 section(s) while group in '{{.*}}1.o' has 2 section(s)
// REVERSE: warning: comdat group with signature 'func' in '{{.*}}1.o' has 2 section(s) while group in '{{.*}}0.o' has 1 section(s)

  .global func
  .section .data.func,"awG",@progbits,func,comdat
func:
  .word 7

  .global func2
  .section .data.func2,"awG",@progbits,func,comdat
func2:
  .word 7
