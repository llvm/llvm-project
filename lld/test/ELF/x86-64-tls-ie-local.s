# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s

# SEC: .got PROGBITS 00000000000020b0 0020b0 000010 00 WA 0 0 8

## Dynamic relocations for non-preemptable symbols in a shared object have section index 0.
# REL:      .rela.dyn {
# REL-NEXT:   0x20B0 R_X86_64_TPOFF64 - 0x0
# REL-NEXT:   0x20B8 R_X86_64_TPOFF64 - 0x4
# REL-NEXT: }

## &.got[0] - 0x1007 = 0x20B0 - 0x1007 = 4265
## &.got[1] - 0x100e = 0x20B8 - 0x100e = 4266
# CHECK:      1000:       addq 4265(%rip), %rax
# CHECK-NEXT: 1007:       addq 4266(%rip), %rax

addq foo@GOTTPOFF(%rip), %rax
addq bar@GOTTPOFF(%rip), %rax

.section .tbss,"awT",@nobits
foo:
  .long 0
bar:
  .long 0
