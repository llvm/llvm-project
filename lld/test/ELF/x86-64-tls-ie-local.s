# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s

# SEC: .got PROGBITS 0000000000002378 000378 000020 00 WA 0 0 8

## Dynamic relocations for non-preemptable symbols in a shared object have section index 0.
# REL:      .rela.dyn {
# REL-NEXT:   0x2378 R_X86_64_TPOFF64 - 0x0
# REL-NEXT:   0x2380 R_X86_64_TPOFF64 - 0x8
# REL-NEXT:   0x2388 R_X86_64_TPOFF64 - 0x4
# REL-NEXT:   0x2390 R_X86_64_TPOFF64 - 0xC
# REL-NEXT: }

## &.got[0] - 0x12af = 0x2378 - 0x12af = 4297
## &.got[1] - 0x12b6 = 0x2380 - 0x12b6 = 4298
## &.got[2] - 0x12be = 0x2388 - 0x12be = 4298
## &.got[3] - 0x12c6 = 0x2390 - 0x12c6 = 4298

# CHECK:      12a8:       addq 4297(%rip), %rax
# CHECK-NEXT: 12af:       addq 4298(%rip), %rax
# CHECK-NEXT: 12b6:       addq 4298(%rip), %r16
# CHECK-NEXT: 12be:       addq 4298(%rip), %r16

addq foo@GOTTPOFF(%rip), %rax
addq bar@GOTTPOFF(%rip), %rax
addq foo2@GOTTPOFF(%rip), %r16
addq bar2@GOTTPOFF(%rip), %r16


.section .tbss,"awT",@nobits
foo:
  .long 0
foo2:
  .long 0
bar:
  .long 0
bar2:
  .long 0
