# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-pc-linux %S/Inputs/gotpc-relax-und-dso.s -o %tdso.o
# RUN: ld.lld -shared %tdso.o -o %t.so
# RUN: ld.lld -shared %t.o %t.so -o %tout
# RUN: llvm-readobj -r -s %tout | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %tout | FileCheck --check-prefix=DISASM %s

# RELOC:      Relocations [
# RELOC-NEXT:   Section ({{.*}}) .rela.dyn {
# RELOC-NEXT:     0x20A8 R_X86_64_GLOB_DAT dsofoo 0x0
# RELOC-NEXT:     0x20B0 R_X86_64_GLOB_DAT foo 0x0
# RELOC-NEXT:     0x20A0 R_X86_64_GLOB_DAT und 0x0
# RELOC-NEXT:   }
# RELOC-NEXT: ]

# 0x101e + 7 - 36 = 0x1001
# 0x1025 + 7 - 43 = 0x1001
# DISASM:      Disassembly of section .text:
# DISASM-NEXT: foo:
# DISASM-NEXT:     1000: 90 nop
# DISASM:      hid:
# DISASM-NEXT:     1001: 90 nop
# DISASM:      _start:
# DISASM-NEXT:    1002: 48 8b 05 97 10 00 00    movq    4247(%rip), %rax
# DISASM-NEXT:    1009: 48 8b 05 90 10 00 00    movq    4240(%rip), %rax
# DISASM-NEXT:    1010: 48 8b 05 91 10 00 00    movq    4241(%rip), %rax
# DISASM-NEXT:    1017: 48 8b 05 8a 10 00 00    movq    4234(%rip), %rax
# DISASM-NEXT:    101e: 48 8d 05 dc ff ff ff    leaq    -36(%rip), %rax
# DISASM-NEXT:    1025: 48 8d 05 d5 ff ff ff    leaq    -43(%rip), %rax
# DISASM-NEXT:    102c: 48 8b 05 7d 10 00 00    movq    4221(%rip), %rax
# DISASM-NEXT:    1033: 48 8b 05 76 10 00 00    movq    4214(%rip), %rax
# DISASM-NEXT:    103a: 8b 05 60 10 00 00       movl    4192(%rip), %eax
# DISASM-NEXT:    1040: 8b 05 5a 10 00 00       movl    4186(%rip), %eax
# DISASM-NEXT:    1046: 8b 05 5c 10 00 00       movl    4188(%rip), %eax
# DISASM-NEXT:    104c: 8b 05 56 10 00 00       movl    4182(%rip), %eax
# DISASM-NEXT:    1052: 8d 05 a9 ff ff ff       leal    -87(%rip), %eax
# DISASM-NEXT:    1058: 8d 05 a3 ff ff ff       leal    -93(%rip), %eax
# DISASM-NEXT:    105e: 8b 05 4c 10 00 00       movl    4172(%rip), %eax
# DISASM-NEXT:    1064: 8b 05 46 10 00 00       movl    4166(%rip), %eax

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

.globl _start
.type _start, @function
_start:
 movq und@GOTPCREL(%rip), %rax
 movq und@GOTPCREL(%rip), %rax
 movq dsofoo@GOTPCREL(%rip), %rax
 movq dsofoo@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq hid@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movq foo@GOTPCREL(%rip), %rax
 movl und@GOTPCREL(%rip), %eax
 movl und@GOTPCREL(%rip), %eax
 movl dsofoo@GOTPCREL(%rip), %eax
 movl dsofoo@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl hid@GOTPCREL(%rip), %eax
 movl foo@GOTPCREL(%rip), %eax
 movl foo@GOTPCREL(%rip), %eax
