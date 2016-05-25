# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# RELOC:    Relocations [
# RELOC:    ]

# 0x11003 + 7 - 10 = 0x11000
# 0x1100a + 7 - 17 = 0x11000
# 0x11011 + 7 - 23 = 0x11001
# 0x11018 + 7 - 30 = 0x11001
# DISASM:      Disassembly of section .text:
# DISASM-NEXT: foo:
# DISASM-NEXT:   11000: 90 nop
# DISASM:      hid:
# DISASM-NEXT:   11001: 90 nop
# DISASM:      ifunc:
# DISASM-NEXT:   11002: c3 retq
# DISASM:      _start:
# DISASM-NEXT: 11003: 48 8d 05 f6 ff ff ff leaq -10(%rip), %rax
# DISASM-NEXT: 1100a: 48 8d 05 ef ff ff ff leaq -17(%rip), %rax
# DISASM-NEXT: 11011: 48 8d 05 e9 ff ff ff leaq -23(%rip), %rax
# DISASM-NEXT: 11018: 48 8d 05 e2 ff ff ff leaq -30(%rip), %rax
# DISASM-NEXT: 1101f: 48 8b 05 da 0f 00 00 movq 4058(%rip), %rax
# DISASM-NEXT: 11026: 48 8b 05 d3 0f 00 00 movq 4051(%rip), %rax
# DISASM-NEXT: 1102d: 8d 05 cd ff ff ff    leal -51(%rip), %eax
# DISASM-NEXT: 11033: 8d 05 c7 ff ff ff    leal -57(%rip), %eax
# DISASM-NEXT: 11039: 8d 05 c2 ff ff ff    leal -62(%rip), %eax
# DISASM-NEXT: 1103f: 8d 05 bc ff ff ff    leal -68(%rip), %eax
# DISASM-NEXT: 11045: 8b 05 b5 0f 00 00    movl 4021(%rip), %eax
# DISASM-NEXT: 1104b: 8b 05 af 0f 00 00    movl 4015(%rip), %eax
# DISASM-NEXT: 11051: 67 e8 a9 ff ff ff    callq -87 <foo>
# DISASM-NEXT: 11057: 67 e8 a3 ff ff ff    callq -93 <foo>
# DISASM-NEXT: 1105d: 67 e8 9e ff ff ff    callq -98 <hid>
# DISASM-NEXT: 11063: 67 e8 98 ff ff ff    callq -104 <hid>
# DISASM-NEXT: 11069: ff 15 91 0f 00 00    callq *3985(%rip)
# DISASM-NEXT: 1106f: ff 15 8b 0f 00 00    callq *3979(%rip)
# DISASM-NEXT: 11075: e9 86 ff ff ff       jmp   -122 <foo>
# DISASM-NEXT: 1107a: 90                   nop
# DISASM-NEXT: 1107b: e9 80 ff ff ff       jmp   -128 <foo>
# DISASM-NEXT: 11080: 90                   nop
# DISASM-NEXT: 11081: e9 7b ff ff ff       jmp   -133 <hid>
# DISASM-NEXT: 11086: 90                   nop
# DISASM-NEXT: 11087: e9 75 ff ff ff       jmp   -139 <hid>
# DISASM-NEXT: 1108c: 90                   nop
# DISASM-NEXT: 1108d: ff 25 6d 0f 00 00    jmpq  *3949(%rip)
# DISASM-NEXT: 11093: ff 25 67 0f 00 00    jmpq  *3943(%rip)

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
 movl foo@GOTPCREL(%rip), %eax
 movl foo@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl ifunc@GOTPCREL(%rip), %eax
 movl ifunc@GOTPCREL(%rip), %eax

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
