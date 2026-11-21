# REQUIRES: x86
## R_386_TLS_IE in a -shared link need R_386_RELATIVE dynamic relocations and GOT slots need R_386_TLS_TPOFF.
## Two input files test ensure there is no race caught by ThreadSanitizer.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/tls-opt-iele-i686-nopic.s -o so.o
# RUN: ld.lld -shared -soname=t.so so.o -o t.so
# RUN: ld.lld -shared a.o b.o t.so -o out
# RUN: llvm-readelf -S -r -d out | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn out | FileCheck --check-prefix=DIS %s

# CHECK:      .got PROGBITS 00003388 000388 000010 00 WA 0 0 4
# CHECK:      0x6ffffffa (RELCOUNT) 8
# CHECK:      Relocation section '.rel.dyn' at offset {{.*}} contains 12 entries:
# CHECK-NEXT:  Offset     Info    Type                Sym. Value  Symbol's Name
# CHECK-NEXT: 000022da 00000008 R_386_RELATIVE
# CHECK-NEXT: 000022e2 00000008 R_386_RELATIVE
# CHECK-NEXT: 000022eb 00000008 R_386_RELATIVE
# CHECK-NEXT: 000022f4 00000008 R_386_RELATIVE
# CHECK-NEXT: 000022fc 00000008 R_386_RELATIVE
# CHECK-NEXT: 00002305 00000008 R_386_RELATIVE
# CHECK-NEXT: 0000230e 00000008 R_386_RELATIVE
# CHECK-NEXT: 00002317 00000008 R_386_RELATIVE
# CHECK-NEXT: 00003390 0000010e R_386_TLS_TPOFF 00000000 tlsshared0
# CHECK-NEXT: 00003394 0000020e R_386_TLS_TPOFF 00000000 tlsshared1
# CHECK-NEXT: 00003388 0000030e R_386_TLS_TPOFF 00000000 tlslocal0
# CHECK-NEXT: 0000338c 0000040e R_386_TLS_TPOFF 00000004 tlslocal1

# DIS:       Disassembly of section test:
# DIS-EMPTY:
# DIS-NEXT:  <_start>:
## (.got)[0] = 0x3388 = 13192
## (.got)[1] = 13196
## (.got)[2] = 13200
## (.got)[3] = 13204
# DIS-NEXT:              movl  13192, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax
# DIS-NEXT:              movl  13192, %eax
# DIS-NEXT:              movl  %gs:(%eax), %eax
# DIS-NEXT:              addl  13192, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax
# DIS-NEXT:              movl  13196, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax
# DIS-NEXT:              movl  13196, %eax
# DIS-NEXT:              movl  %gs:(%eax), %eax
# DIS-NEXT:              addl  13196, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax
# DIS-NEXT:              movl  13200, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax
# DIS-NEXT:              addl  13204, %ecx
# DIS-NEXT:              movl  %gs:(%ecx), %eax

#--- a.s
.type tlslocal0,@object
.section .tbss,"awT",@nobits
.globl tlslocal0
.align 4
tlslocal0:
 .long 0
 .size tlslocal0, 4

.type tlslocal1,@object
.section .tbss,"awT",@nobits
.globl tlslocal1
.align 4
tlslocal1:
 .long 0
 .size tlslocal1, 4

.section .text
.globl ___tls_get_addr
.type ___tls_get_addr,@function
___tls_get_addr:

.section test, "axw"
.globl _start
_start:
movl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal0@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

#--- b.s
.section test, "axw"
movl tlslocal1@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlsshared0@indntpoff,%ecx
movl %gs:(%ecx),%eax

addl tlsshared1@indntpoff,%ecx
movl %gs:(%ecx),%eax
