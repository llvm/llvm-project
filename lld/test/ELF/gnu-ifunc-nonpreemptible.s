# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DISASM
# RUN: llvm-readelf -r -s %t | FileCheck %s

# RUN: ld.lld --export-dynamic %t.o -o %t
# RUN: llvm-readelf -r -s %t | FileCheck %s

# CHECK:      Relocation section '.rela.dyn' at offset {{.*}} contains 2 entries:
# CHECK-NEXT:     Type
# CHECK-NEXT: R_X86_64_IRELATIVE
# CHECK-NEXT: R_X86_64_IRELATIVE

# CHECK:      0 NOTYPE  LOCAL  HIDDEN     [[#]] __rela_iplt_start
# CHECK-NEXT: 0 NOTYPE  LOCAL  HIDDEN     [[#]] __rela_iplt_end

# RUN: ld.lld -pie %t.o -o %t1
# RUN: llvm-readelf -s %t1 | FileCheck %s --check-prefix=PIC
# RUN: ld.lld -shared %t.o -o %t2
# RUN: llvm-readelf -s %t2 | FileCheck %s --check-prefix=PIC

# PIC:        0 NOTYPE  WEAK   DEFAULT    UND __rela_iplt_start
# PIC-NEXT:   0 NOTYPE  WEAK   DEFAULT    UND __rela_iplt_end

# DISASM: Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <foo>:
# DISASM:      <bar>:
# DISASM:      <unused>:
# DISASM:      <_start>:
# DISASM-NEXT:   callq 0x[[#%x,foo:]]
# DISASM-NEXT:   callq 0x[[#%x,bar:]]
# DISASM-EMPTY:
# DISASM-NEXT: Disassembly of section .iplt:
# DISASM-EMPTY:
# DISASM-NEXT: <.iplt>:
# DISASM-NEXT:  [[#foo]]: jmpq *{{.*}}(%rip)
# DISASM-NEXT:            pushq $0
# DISASM-NEXT:            jmp 0x0
# DISASM-NEXT:  [[#bar]]: jmpq *{{.*}}(%rip)
# DISASM-NEXT:            pushq $1
# DISASM-NEXT:            jmp 0x0

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.type unused, @gnu_indirect_function
.globl unused
unused:
  ret

.weak __rela_iplt_start
.weak __rela_iplt_end

.globl _start
_start:
 call foo
 call bar

.data
  .quad __rela_iplt_start
  .quad __rela_iplt_end
