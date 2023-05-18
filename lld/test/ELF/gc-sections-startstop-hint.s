# REQUIRES: x86
## Some projects may not work with GNU ld<2015-10 (ld.lld 13.0.0) --gc-sections behavior.
## Give a hint.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null
# RUN: ld.lld %t.o --gc-sections -z nostart-stop-gc -o /dev/null
# RUN: not ld.lld %t.o --gc-sections -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: __start_meta
# CHECK-NEXT: >>> referenced by {{.*}}
# CHECK-NEXT: >>> the encapsulation symbol needs to be retained under --gc-sections properly; consider -z nostart-stop-gc (see https://lld.llvm.org/ELF/start-stop-gc)

.section .text,"ax",@progbits
.global _start
_start:
  .quad __start_meta - .
  .quad __stop_meta - .

.section meta,"aw",@progbits
.quad 0
