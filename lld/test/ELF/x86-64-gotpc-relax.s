# REQUIRES: x86
## Test R_X86_64_GOTPCRELX and R_X86_64_REX_GOTPCRELX GOT optimization.

# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1 --no-apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,NOAPPLY %s
# RUN: ld.lld %t.o -o %t1 --apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,APPLY %s
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefix=DISASM %s

## --no-relax disables GOT optimization.
# RUN: ld.lld --no-relax %t.o -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck --check-prefix=NORELAX %s

## In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# CHECK:      Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .iplt             PROGBITS        0000000000201280 000280 000010 00  AX  0   0 16
# CHECK-NEXT: .got              PROGBITS        0000000000202290 000290 000000 00  WA  0   0  8

## There is one R_X86_64_IRELATIVE relocations.
# RELOC-LABEL: Relocation section '.rela.dyn' at offset {{.*}} contains 1 entry:
# CHECK:           Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK:       0000000000203290  0000000000000025 R_X86_64_IRELATIVE                        2011e2
# CHECK-LABEL: Hex dump of section '.got.plt':
# NOAPPLY-NEXT:  0x00203290 00000000 00000000
# APPLY-NEXT:    0x00203290 e2112000 00000000

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
# DISASM-NEXT: movq 8330(%rip), %rax
# DISASM-NEXT: movq 8323(%rip), %rax
# DISASM-NEXT: leaq -52(%rip), %rax
# DISASM-NEXT: leaq -59(%rip), %rax
# DISASM-NEXT: leaq -65(%rip), %rax
# DISASM-NEXT: leaq -72(%rip), %rax
# DISASM-NEXT: movq 8288(%rip), %rax
# DISASM-NEXT: movq 8281(%rip), %rax
# DISASM-NEXT: callq 0x2011e0 <foo>
# DISASM-NEXT: callq 0x2011e0 <foo>
# DISASM-NEXT: callq 0x2011e1 <hid>
# DISASM-NEXT: callq 0x2011e1 <hid>
# DISASM-NEXT: callq *8251(%rip)
# DISASM-NEXT: callq *8245(%rip)
# DISASM-NEXT: jmp   0x2011e0 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e0 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e1 <hid>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x2011e1 <hid>
# DISASM-NEXT: nop
# DISASM-NEXT: jmpq  *8215(%rip)
# DISASM-NEXT: jmpq  *8209(%rip)

# NORELAX-LABEL: <_start>:
# NORELAX-COUNT-12: movq
# NORELAX-COUNT-6:  callq *
# NORELAX-COUNT-6:  jmpq *

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
