# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj --sections -r %t.so | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=DIS

.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
.align 4
tls0:
 .long 0
 .size tls0, 4

.type  tls1,@object
.globl tls1
.align 4
tls1:
 .long 0
 .size tls1, 4

.type  tls2,@object
.globl tls2
.hidden tls2
.align 4
tls2:
 .long 0
 .size tls2, 8

.section .text
.globl _start
_start:
leal tls0@tlsgd(,%ebx,1),%eax
call __tls_get_addr@plt

leal tls1@tlsgd(,%ebx,1),%eax
call __tls_get_addr@plt

leal tls2@tlsldm(%ebx),%eax
call __tls_get_addr@plt
leal tls2@dtpoff(%eax),%edx

leal tls2@tlsldm(%ebx),%eax
call __tls_get_addr@plt
leal tls2@dtpoff+4(%eax),%edx

movl %gs:0,%eax
addl tls0@gotntpoff(%ebx),%eax

movl %gs:0,%eax
addl tls1@gotntpoff(%ebx),%eax

# CHECK:      Name: .got (
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT:   SHF_WRITE
# CHECK-NEXT: ]
# CHECK-NEXT: Address: 0x2358
# CHECK-NEXT: Offset: 0x358
# CHECK-NEXT: Size: 32
# CHECK-NEXT: Link: 0
# CHECK-NEXT: Info: 0
# CHECK-NEXT: AddressAlignment: 4
# CHECK-NEXT: EntrySize: 0

# CHECK: Relocations [
# CHECK:      Section ({{.+}}) .rel.dyn {
# CHECK-NEXT: 0x2370 R_386_TLS_DTPMOD32 -
# CHECK-NEXT: 0x2358 R_386_TLS_DTPMOD32 tls0
# CHECK-NEXT: 0x235C R_386_TLS_DTPOFF32 tls0
# CHECK-NEXT: 0x2360 R_386_TLS_TPOFF tls0
# CHECK-NEXT: 0x2364 R_386_TLS_DTPMOD32 tls1
# CHECK-NEXT: 0x2368 R_386_TLS_DTPOFF32 tls1
# CHECK-NEXT: 0x236C R_386_TLS_TPOFF tls1
# CHECK-NEXT: }

# DIS:      Disassembly of section .text:
# DIS-EMPTY:
# DIS-NEXT: <_start>:
## General dynamic model:
## -4128 and -4116 are first and second GOT entries offsets.
## Each one is a pair of records.
# DIS-NEXT: 1260:       leal -4128(,%ebx), %eax
# DIS-NEXT: 1267:       calll 0x12d0
# DIS-NEXT: 126c:       leal -4116(,%ebx), %eax
# DIS-NEXT: 1273:       calll 0x12d0
## Local dynamic model:
## -16 is a local module tls index offset.
# DIS-NEXT: 1278:       leal -4104(%ebx), %eax
# DIS-NEXT: 127e:       calll 0x12d0
# DIS-NEXT: 1283:       leal 8(%eax), %edx
# DIS-NEXT: 1289:       leal -4104(%ebx), %eax
# DIS-NEXT: 128f:       calll 0x12d0
# DIS-NEXT: 1294:       leal 12(%eax), %edx
## Initial exec model:
# DIS-NEXT: 129a:       movl %gs:0, %eax
# DIS-NEXT: 12a0:       addl -4120(%ebx), %eax
# DIS-NEXT: 12a6:       movl %gs:0, %eax
# DIS-NEXT: 12ac:       addl -4108(%ebx), %eax
