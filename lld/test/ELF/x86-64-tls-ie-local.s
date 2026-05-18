# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s

# SEC: .got PROGBITS 0000000000002378 000378 000010 00 WA 0 0 8

## Dynamic relocations for non-preemptable symbols in a shared object have section index 0.
# REL:      .rela.dyn {
# REL-NEXT:   0x2378 R_X86_64_TPOFF64 - 0x0
# REL-NEXT:   0x2380 R_X86_64_TPOFF64 - 0x4
# REL-NEXT: }

## &.got[0] - 0x127f = 0x2378 - 0x127f = 4345
## &.got[1] - 0x1286 = 0x2380 - 0x1286 = 4346
## &.got[2] - 0x128e = 0x2378 - 0x128e = 4330
## &.got[3] - 0x1296 = 0x2380 - 0x1296 = 4330
## &.got[0] - 0x12a0 = 0x2378 - 0x12a0 = 4312
## &.got[1] - 0x12aa = 0x2380 - 0x12aa = 4310
## &.got[0] - 0x12b4 = 0x2378 - 0x12b4 = 4292
## &.got[1] - 0x12be = 0x2380 - 0x12be = 4290
## &.got[0] - 0x12c8 = 0x2378 - 0x12c8 = 4272

# CHECK:      1278:       addq 4345(%rip), %rax
# CHECK-NEXT: 127f:       addq 4346(%rip), %rax
# CHECK-NEXT: 1286:       addq 4330(%rip), %r16
# CHECK-NEXT: 128e:       addq 4330(%rip), %r16
# CHECK-NEXT: 1296:       addq %r8, 4312(%rip), %r16
# CHECK-NEXT: 12a0:       addq 4310(%rip), %rax, %r12
# CHECK-NEXT: 12aa:       {nf} addq %r8, 4292(%rip), %r16
# CHECK-NEXT: 12b4:       {nf} addq 4290(%rip), %rax, %r12
# CHECK-NEXT: 12be:       {nf} addq 4272(%rip), %r12

addq foo@GOTTPOFF(%rip), %rax
addq bar@GOTTPOFF(%rip), %rax
# EGPR
addq foo@GOTTPOFF(%rip), %r16
addq bar@GOTTPOFF(%rip), %r16
# NDD
addq %r8, foo@GOTTPOFF(%rip), %r16
addq bar@GOTTPOFF(%rip), %rax, %r12
# NDD + NF
{nf} addq %r8, foo@GOTTPOFF(%rip), %r16
{nf} addq bar@GOTTPOFF(%rip), %rax, %r12
# NF
{nf} addq foo@GOTTPOFF(%rip), %r12

.section .tbss,"awT",@nobits
foo:
  .long 0
bar:
  .long 0
