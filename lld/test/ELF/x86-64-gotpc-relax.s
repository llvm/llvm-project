# REQUIRES: x86
## Test R_X86_64_GOTPCRELX and R_X86_64_REX_GOTPCRELX/R_X86_64_CODE_4_GOTPCRELX GOT optimization.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o -x86-apx-relax-relocations=true
# RUN: ld.lld %t.o -o %t1 --no-apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,APXRELAX,NOAPPLY-APXRELAX %s
# RUN: ld.lld %t.o -o %t1 --apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,APPLY-APXRELAX %s
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefixes=DISASM,DISASM-APXRELAX %s

## --no-relax disables GOT optimization.
# RUN: ld.lld --no-relax %t.o -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck --check-prefix=NORELAX %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1 --no-apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,NOAPXRELAX,NOAPPLY-NOAPXRELAX %s
# RUN: ld.lld %t.o -o %t1 --apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,APPLY-NOAPXRELAX %s
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefixes=DISASM,DISASM-NOAPXRELAX %s

## --no-relax disables GOT optimization.
# RUN: ld.lld --no-relax %t.o -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck --check-prefix=NORELAX %s

## In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# CHECK:      Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .iplt             PROGBITS        00000000002012e0 0002e0 000010 00  AX  0   0 16
# APXRELAX-NEXT: .got              PROGBITS        00000000002022f0 0002f0 000000 00  WA  0   0  8
# NOAPXRELAX-NEXT: .got              PROGBITS        00000000002022f0 0002f0 000010 00  WA  0   0  8

## There is one R_X86_64_IRELATIVE relocations.
# RELOC-LABEL: Relocation section '.rela.dyn' at offset {{.*}} contains 1 entry:
# CHECK:           Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# APXRELAX:       00000000002032f0  0000000000000025 R_X86_64_IRELATIVE                        2011e2
# NOAPXRELAX:       0000000000203300  0000000000000025 R_X86_64_IRELATIVE                        2011e2
# CHECK-LABEL: Hex dump of section '.got.plt':
# NOAPPLY-APXRELAX-NEXT:  0x002032f0 00000000 00000000
# NOAPPLY-NOAPXRELAX-NEXT:  0x00203300 00000000 00000000
# APPLY-APXRELAX-NEXT:    0x002032f0 e2112000 00000000
# APPLY-NOAPXRELAX-NEXT:    0x00203300 e2112000 00000000

# 0x201173 + 7 - 10 = 0x201170
# 0x20117a + 7 - 17 = 0x201170
# 0x201181 + 7 - 23 = 0x201171
# 0x201188 + 7 - 30 = 0x201171
# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <foo>:
# DISASM-NEXT:   2011e0: 90 nop
# DISASM:      <hid>:
# DISASM-NEXT:   2011e1: 90 nop
# DISASM:      <ifunc>:
# DISASM-NEXT:   2011e2: c3 retq
# DISASM:      <_start>:
# DISASM-NEXT: leaq -10(%rip), %rax
# DISASM-NEXT: leaq -17(%rip), %rax
# DISASM-NEXT: leaq -23(%rip), %rax
# DISASM-NEXT: leaq -30(%rip), %rax
# DISASM-APXRELAX-NEXT: movq 8426(%rip), %rax
# DISASM-APXRELAX-NEXT: movq 8419(%rip), %rax
# DISASM-NOAPXRELAX-NEXT: movq 8442(%rip), %rax
# DISASM-NOAPXRELAX-NEXT: movq 8435(%rip), %rax
# DISASM-NEXT: leaq -52(%rip), %rax
# DISASM-NEXT: leaq -59(%rip), %rax
# DISASM-NEXT: leaq -65(%rip), %rax
# DISASM-NEXT: leaq -72(%rip), %rax
# DISASM-APXRELAX-NEXT: movq 8384(%rip), %rax
# DISASM-APXRELAX-NEXT: movq 8377(%rip), %rax
# DISASM-NOAPXRELAX-NEXT: movq 8400(%rip), %rax
# DISASM-NOAPXRELAX-NEXT: movq 8393(%rip), %rax
# DISASM-NEXT: callq 0x2011e0 <foo>
# DISASM-NEXT: callq 0x2011e0 <foo>
# DISASM-NEXT: callq 0x2011e1 <hid>
# DISASM-NEXT: callq 0x2011e1 <hid>
# DISASM-APXRELAX-NEXT: callq *8347(%rip)
# DISASM-APXRELAX-NEXT: callq *8341(%rip)
# DISASM-NOAPXRELAX-NEXT: callq *8363(%rip)
# DISASM-NOAPXRELAX-NEXT: callq *8357(%rip)
# DISASM-NEXT: jmp   0x2011e0 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e0 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e1 <hid>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e1 <hid>
# DISASM-NEXT: nop
# DISASM-APXRELAX-NEXT: jmpq  *8311(%rip)
# DISASM-APXRELAX-NEXT: jmpq  *8305(%rip)
# DISASM-NOAPXRELAX-NEXT: jmpq *8327(%rip)
# DISASM-NOAPXRELAX-NEXT: jmpq *8321(%rip)
# DISASM-APXRELAX-NEXT: leaq -167(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -175(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -182(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -190(%rip), %r16
# DISASM-APXRELAX-NEXT: movq 8265(%rip), %r16
# DISASM-APXRELAX-NEXT: movq 8257(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -215(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -223(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -230(%rip), %r16
# DISASM-APXRELAX-NEXT: leaq -238(%rip), %r16
# DISASM-APXRELAX-NEXT: movq 8217(%rip), %r16
# DISASM-APXRELAX-NEXT: movq 8209(%rip), %r16

# DISASM-NOAPXRELAX-NEXT: movq 4201(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4193(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4193(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4185(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 8281(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 8273(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4153(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4145(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4145(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 4137(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 8233(%rip), %r16
# DISASM-NOAPXRELAX-NEXT: movq 8225(%rip), %r16

# NORELAX-LABEL: <_start>:
# NORELAX-COUNT-12: movq
# NORELAX-COUNT-6:  callq *
# NORELAX-COUNT-6:  jmpq *
# NORELAX-COUNT-12: movq

.text
.globl foo
.type foo, @function
foo:
 nop

.globl hid
.hidden hid
.type hid, @function
hid:
 nop

.text
.type ifunc STT_GNU_IFUNC
.globl ifunc
.type ifunc, @function
ifunc:
 ret

.globl _start
.type _start, @function
_start:
 movq foo@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax
 movq ifunc@GOTPCREL(%rip), %rax

 call *foo@GOTPCREL(%rip)
 call *foo@GOTPCREL(%rip)
 call *hid@GOTPCREL(%rip)
 call *hid@GOTPCREL(%rip)
 call *ifunc@GOTPCREL(%rip)
 call *ifunc@GOTPCREL(%rip)
 jmp *foo@GOTPCREL(%rip)
 jmp *foo@GOTPCREL(%rip)
 jmp *hid@GOTPCREL(%rip)
 jmp *hid@GOTPCREL(%rip)
 jmp *ifunc@GOTPCREL(%rip)
 jmp *ifunc@GOTPCREL(%rip)

 movq foo@GOTPCREL(%rip), %r16
 movq foo@GOTPCREL(%rip), %r16
 movq hid@GOTPCREL(%rip), %r16
 movq hid@GOTPCREL(%rip), %r16
 movq ifunc@GOTPCREL(%rip), %r16
 movq ifunc@GOTPCREL(%rip), %r16
 movq foo@GOTPCREL(%rip), %r16
 movq foo@GOTPCREL(%rip), %r16
 movq hid@GOTPCREL(%rip), %r16
 movq hid@GOTPCREL(%rip), %r16
 movq ifunc@GOTPCREL(%rip), %r16
 movq ifunc@GOTPCREL(%rip), %r16
