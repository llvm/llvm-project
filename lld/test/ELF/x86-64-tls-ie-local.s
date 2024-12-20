# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s

# SEC: .got PROGBITS 0000000000002348 000348 000010 00 WA 0 0 8

## Dynamic relocations for non-preemptable symbols in a shared object have section index 0.
# REL:      .rela.dyn {
# REL-NEXT:   0x2348 R_X86_64_TPOFF64 - 0x0
# REL-NEXT:   0x2350 R_X86_64_TPOFF64 - 0x4
# REL-NEXT: }

## &.got[0] - 0x127f = 0x2348 - 0x127f = 4297
## &.got[1] - 0x1286 = 0x2350 - 0x1286 = 4298
## &.got[2] - 0x128e = 0x2348 - 0x128e = 4282
## &.got[3] - 0x1296 = 0x2350 - 0x1296 = 4282

# CHECK:      1278:       addq 4297(%rip), %rax
# CHECK-NEXT: 127f:       addq 4298(%rip), %rax
# CHECK-NEXT: 1286:       addq 4282(%rip), %r16
# CHECK-NEXT: 128e:       addq 4282(%rip), %r16

addq foo@GOTTPOFF(%rip), %rax
addq bar@GOTTPOFF(%rip), %rax
addq foo@GOTTPOFF(%rip), %r16
addq bar@GOTTPOFF(%rip), %r16


.section .tbss,"awT",@nobits
foo:
  .long 0
bar:
  .long 0
