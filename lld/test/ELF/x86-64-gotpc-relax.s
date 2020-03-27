# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# RELOC:    Relocations [
# RELOC:    ]

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
