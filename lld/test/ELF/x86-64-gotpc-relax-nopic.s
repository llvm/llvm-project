# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --hash-style=sysv %t.o -o %t1
# RUN: llvm-readobj --symbols -r %t1 | FileCheck --check-prefix=SYMRELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# SYMRELOC:      Relocations [
# SYMRELOC-NEXT: ]
# SYMRELOC:      Symbols [
# SYMRELOC:       Symbol {
# SYMRELOC:        Name: bar
# SYMRELOC-NEXT:   Value: 0x202197

## 2105751 = 0x202197 (bar)
# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: _start:
# DISASM-NEXT:   201158:       adcq  $2105751, %rax
# DISASM-NEXT:                 addq  $2105751, %rbx
# DISASM-NEXT:                 andq  $2105751, %rcx
# DISASM-NEXT:                 cmpq  $2105751, %rdx
# DISASM-NEXT:                 orq   $2105751, %rdi
# DISASM-NEXT:                 sbbq  $2105751, %rsi
# DISASM-NEXT:                 subq  $2105751, %rbp
# DISASM-NEXT:                 xorq  $2105751, %r8
# DISASM-NEXT:                 testq $2105751, %r15

# RUN: ld.lld --hash-style=sysv -shared %t.o -o %t2
# RUN: llvm-readobj -S -r -d %t2 | FileCheck --check-prefix=SEC-PIC    %s
# RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck --check-prefix=DISASM-PIC %s
# SEC-PIC:      Section {
# SEC-PIC:        Index:
# SEC-PIC:        Name: .got
# SEC-PIC-NEXT:   Type: SHT_PROGBITS
# SEC-PIC-NEXT:   Flags [
# SEC-PIC-NEXT:     SHF_ALLOC
# SEC-PIC-NEXT:     SHF_WRITE
# SEC-PIC-NEXT:   ]
# SEC-PIC-NEXT:   Address: 0x2348
# SEC-PIC-NEXT:   Offset: 0x348
# SEC-PIC-NEXT:   Size: 8
# SEC-PIC-NEXT:   Link:
# SEC-PIC-NEXT:   Info:
# SEC-PIC-NEXT:   AddressAlignment:
# SEC-PIC-NEXT:   EntrySize:
# SEC-PIC-NEXT: }
# SEC-PIC:      Relocations [
# SEC-PIC-NEXT:   Section ({{.*}}) .rela.dyn {
# SEC-PIC-NEXT:     0x2348 R_X86_64_RELATIVE - 0x3350
# SEC-PIC-NEXT:   }
# SEC-PIC-NEXT: ]
# SEC-PIC:      0x000000006FFFFFF9 RELACOUNT            1

## Check that there was no relaxation performed. All values refer to got entry.
## Ex: 0x1000 + 4249 + 7 = 0x20A0
##     0x102a + 4207 + 7 = 0x20A0
# DISASM-PIC:      Disassembly of section .text:
# DISASM-PIC-EMPTY:
# DISASM-PIC-NEXT: _start:
# DISASM-PIC-NEXT: 1268:       adcq  4313(%rip), %rax
# DISASM-PIC-NEXT:             addq  4306(%rip), %rbx
# DISASM-PIC-NEXT:             andq  4299(%rip), %rcx
# DISASM-PIC-NEXT:             cmpq  4292(%rip), %rdx
# DISASM-PIC-NEXT:             orq   4285(%rip), %rdi
# DISASM-PIC-NEXT:             sbbq  4278(%rip), %rsi
# DISASM-PIC-NEXT:             subq  4271(%rip), %rbp
# DISASM-PIC-NEXT:             xorq  4264(%rip), %r8
# DISASM-PIC-NEXT:             testq %r15, 4257(%rip)

.data
.type   bar, @object
bar:
 .byte   1
 .size   bar, .-bar

.text
.globl  _start
.type   _start, @function
_start:
  adcq    bar@GOTPCREL(%rip), %rax
  addq    bar@GOTPCREL(%rip), %rbx
  andq    bar@GOTPCREL(%rip), %rcx
  cmpq    bar@GOTPCREL(%rip), %rdx
  orq     bar@GOTPCREL(%rip), %rdi
  sbbq    bar@GOTPCREL(%rip), %rsi
  subq    bar@GOTPCREL(%rip), %rbp
  xorq    bar@GOTPCREL(%rip), %r8
  testq   %r15, bar@GOTPCREL(%rip)
