# REQUIRES: x86
## Test R_X86_64_GOTPCRELX and R_X86_64_REX_GOTPCRELX GOT optimization.

# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1 --no-apply-dynamic-relocs
# RUN: llvm-readobj -x .got.plt -r %t1 | FileCheck --check-prefixes=RELOC,NO-APPLY-DYNAMIC-RELOCS %s
# RUN: ld.lld %t.o -o %t1 --apply-dynamic-relocs
# RUN: llvm-readobj -x .got.plt -r %t1 | FileCheck --check-prefixes=RELOC,APPLY-DYNAMIC-RELOCS %s
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

## --no-relax disables GOT optimization.
# RUN: ld.lld --no-relax %t.o -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=NORELAX %s

## There is one R_X86_64_IRELATIVE relocations.
# RELOC-LABEL: Relocations [
# RELOC-NEXT:    Section (1) .rela.dyn {
# RELOC-NEXT:      0x202220 R_X86_64_IRELATIVE - 0x201172
# RELOC-NEXT:    }
# RELOC-NEXT:  ]
# RELOC-LABEL: Hex dump of section '.got.plt':
# NO-APPLY-DYNAMIC-RELOCS-NEXT:  0x00202220 00000000 00000000
# APPLY-DYNAMIC-RELOCS-NEXT:     0x00202220 72112000 00000000
# RELOC-EMPTY:

# 0x201173 + 7 - 10 = 0x201170
# 0x20117a + 7 - 17 = 0x201170
# 0x201181 + 7 - 23 = 0x201171
# 0x201188 + 7 - 30 = 0x201171
# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <foo>:
# DISASM-NEXT:   201170: 90 nop
# DISASM:      <hid>:
# DISASM-NEXT:   201171: 90 nop
# DISASM:      <ifunc>:
# DISASM-NEXT:   201172: c3 retq
# DISASM:      <_start>:
# DISASM-NEXT: leaq -10(%rip), %rax
# DISASM-NEXT: leaq -17(%rip), %rax
# DISASM-NEXT: leaq -23(%rip), %rax
# DISASM-NEXT: leaq -30(%rip), %rax
# DISASM-NEXT: movq 4234(%rip), %rax
# DISASM-NEXT: movq 4227(%rip), %rax
# DISASM-NEXT: leaq -52(%rip), %rax
# DISASM-NEXT: leaq -59(%rip), %rax
# DISASM-NEXT: leaq -65(%rip), %rax
# DISASM-NEXT: leaq -72(%rip), %rax
# DISASM-NEXT: movq 4192(%rip), %rax
# DISASM-NEXT: movq 4185(%rip), %rax
# DISASM-NEXT: callq 0x201170 <foo>
# DISASM-NEXT: callq 0x201170 <foo>
# DISASM-NEXT: callq 0x201171 <hid>
# DISASM-NEXT: callq 0x201171 <hid>
# DISASM-NEXT: callq *4155(%rip)
# DISASM-NEXT: callq *4149(%rip)
# DISASM-NEXT: jmp   0x201170 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x201170 <foo>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x201171 <hid>
# DISASM-NEXT: nop
# DISASM-NEXT: jmp   0x201171 <hid>
# DISASM-NEXT: nop
# DISASM-NEXT: jmpq  *4119(%rip)
# DISASM-NEXT: jmpq  *4113(%rip)

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
