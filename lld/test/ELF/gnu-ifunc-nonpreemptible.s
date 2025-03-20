# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared -soname=b.so b.o -o b.so

# RUN: ld.lld a.o -o a
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn a | FileCheck %s --check-prefix=DISASM
# RUN: llvm-readelf -r -s a | FileCheck %s

# CHECK:      Relocation section '.rela.dyn' at offset {{.*}} contains 3 entries:
# CHECK-NEXT:     Type
# CHECK-NEXT: {{0*}}[[#%x,O:]] [[#%x,]] R_X86_64_IRELATIVE  [[#%x,QUX:]]
# CHECK-NEXT: {{0*}}[[#O+8]]   [[#%x,]] R_X86_64_IRELATIVE
# CHECK-NEXT: {{0*}}[[#O+16]]  [[#%x,]] R_X86_64_IRELATIVE

# CHECK:                      0 NOTYPE  LOCAL  HIDDEN     [[#]] __rela_iplt_start
# CHECK-NEXT:                 0 NOTYPE  LOCAL  HIDDEN     [[#]] __rela_iplt_end
# CHECK-NEXT: {{0*}}[[#QUX]]  0 IFUNC   GLOBAL DEFAULT    [[#]] qux

# RUN: ld.lld -pie a.o b.so -o a1
# RUN: llvm-readelf -rs a1 | FileCheck %s --check-prefixes=PIC,PIE
# RUN: ld.lld -shared a.o b.so -o a2
# RUN: llvm-readelf -rs a2 | FileCheck %s --check-prefix=PIC

# PIC:      {{0*}}[[#%x,O:]] [[#%x,]] R_X86_64_RELATIVE
# PIC-NEXT:                           R_X86_64_GLOB_DAT      0000000000000000 ext + 0
# PIC-NEXT: {{0*}}[[#O-16]]  [[#%x,]] R_X86_64_64            0000000000000000 __rela_iplt_start + 0
# PIC-NEXT: {{0*}}[[#O-8]]   [[#%x,]] R_X86_64_64            0000000000000000 __rela_iplt_end + 0
# PIE-NEXT: {{0*}}[[#O+8]]   [[#%x,]] R_X86_64_IRELATIVE
# PIE-NEXT: {{0*}}[[#O+16]]  [[#%x,]] R_X86_64_IRELATIVE
# PIE-NEXT: {{0*}}[[#O+24]]  [[#%x,]] R_X86_64_IRELATIVE

# PIC:        0 NOTYPE  WEAK   DEFAULT    UND __rela_iplt_start
# PIC-NEXT:   0 NOTYPE  WEAK   DEFAULT    UND __rela_iplt_end

# DISASM: Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <qux>:
# DISASM:      <foo>:
# DISASM:      <bar>:
# DISASM:      <unused>:
# DISASM:      <_start>:
# DISASM-NEXT:   callq 0x[[#%x,foo:]]
# DISASM-NEXT:   callq 0x[[#%x,bar:]]
# DISASM-NEXT:   callq 0x[[#%x,qux:]]
# DISASM-EMPTY:
# DISASM-NEXT: Disassembly of section .iplt:
# DISASM-EMPTY:
# DISASM-NEXT: <.iplt>:
# DISASM-NEXT:  [[#qux]]: jmpq *{{.*}}(%rip)
# DISASM-NEXT:            pushq $0
# DISASM-NEXT:            jmp 0x0
# DISASM-NEXT:  [[#foo]]: jmpq *{{.*}}(%rip)
# DISASM-NEXT:            pushq $1
# DISASM-NEXT:            jmp 0x0
# DISASM-NEXT:  [[#bar]]: jmpq *{{.*}}(%rip)
# DISASM-NEXT:            pushq $2
# DISASM-NEXT:            jmp 0x0

#--- a.s
.globl qux, foo, bar
.type qux, @gnu_indirect_function
.type foo STT_GNU_IFUNC
.type bar STT_GNU_IFUNC
qux: ret
foo: ret
bar: ret

.type unused, @gnu_indirect_function
.globl unused
.weak ext
unused: mov ext@gotpcrel(%rip), %rax

.weak __rela_iplt_start
.weak __rela_iplt_end

.globl _start
_start:
 call foo
 call bar
 call qux

.data
  .quad __rela_iplt_start
  .quad __rela_iplt_end
  .quad .data

#--- b.s
.globl ext
ext:
  ret
