# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s

# SEC: .got PROGBITS 0000000000002338 000338 000010 00 WA 0 0 8

## Dynamic relocations for non-preemptable symbols in a shared object have section index 0.
# REL:      .rela.dyn {
# REL-NEXT:   0x2338 R_X86_64_TPOFF64 - 0x0
# REL-NEXT:   0x2340 R_X86_64_TPOFF64 - 0x4
# REL-NEXT: }

## &.got[0] - 0x127f = 0x2338 - 0x127f = 4281
## &.got[1] - 0x1286 = 0x2340 - 0x1286 = 4282
# CHECK:      1278:       addq 4281(%rip), %rax
# CHECK-NEXT: 127f:       addq 4282(%rip), %rax

addq foo@GOTTPOFF(%rip), %rax
addq bar@GOTTPOFF(%rip), %rax

.section .tbss,"awT",@nobits
foo:
  .long 0
bar:
  .long 0
