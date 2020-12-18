# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -relax-relocations -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj --symbols -r %t1 | FileCheck --check-prefix=SYMRELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t1 | FileCheck --check-prefix=DISASM %s

## There is no relocations.
# SYMRELOC:      Relocations [
# SYMRELOC-NEXT: ]
# SYMRELOC:      Symbols [
# SYMRELOC:       Symbol {
# SYMRELOC:        Name: bar
# SYMRELOC-NEXT:   Value: 0x203248

## 2105751 = 0x202197 (bar)
# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT:   2011c8:       adcl  {{.*}}(%rip), %eax  # 202240
# DISASM-NEXT:                 addl  {{.*}}(%rip), %ebx  # 202240
# DISASM-NEXT:                 andl  {{.*}}(%rip), %ecx  # 202240
# DISASM-NEXT:                 cmpl  {{.*}}(%rip), %edx  # 202240
# DISASM-NEXT:                 orl   {{.*}}(%rip), %edi  # 202240
# DISASM-NEXT:                 sbbl  {{.*}}(%rip), %esi  # 202240
# DISASM-NEXT:                 subl  {{.*}}(%rip), %ebp  # 202240
# DISASM-NEXT:                 xorl  $0x203248, %r8d
# DISASM-NEXT:                 testl $0x203248, %r15d
# DISASM-NEXT:   201200:       adcq  $0x203248, %rax
# DISASM-NEXT:                 addq  $0x203248, %rbx
# DISASM-NEXT:                 andq  $0x203248, %rcx
# DISASM-NEXT:                 cmpq  $0x203248, %rdx
# DISASM-NEXT:                 orq   $0x203248, %rdi
# DISASM-NEXT:                 sbbq  $0x203248, %rsi
# DISASM-NEXT:                 subq  $0x203248, %rbp
# DISASM-NEXT:                 xorq  $0x203248, %r8
# DISASM-NEXT:                 testq $0x203248, %r15

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
# SEC-PIC-NEXT:   Address: 0x2380
# SEC-PIC-NEXT:   Offset: 0x380
# SEC-PIC-NEXT:   Size: 8
# SEC-PIC-NEXT:   Link:
# SEC-PIC-NEXT:   Info:
# SEC-PIC-NEXT:   AddressAlignment:
# SEC-PIC-NEXT:   EntrySize:
# SEC-PIC-NEXT: }
# SEC-PIC:      0x000000006FFFFFF9 RELACOUNT            1
# SEC-PIC:      Relocations [
# SEC-PIC-NEXT:   Section ({{.*}}) .rela.dyn {
# SEC-PIC-NEXT:     0x2380 R_X86_64_RELATIVE - 0x3388
# SEC-PIC-NEXT:   }
# SEC-PIC-NEXT: ]

## Check that there was no relaxation performed. All values refer to got entry.
# DISASM-PIC:      Disassembly of section .text:
# DISASM-PIC-EMPTY:
# DISASM-PIC-NEXT: <_start>:
# DISASM-PIC-NEXT: 1268:       adcl  {{.*}}(%rip), %eax  # 2380
# DISASM-PIC-NEXT:             addl  {{.*}}(%rip), %ebx  # 2380
# DISASM-PIC-NEXT:             andl  {{.*}}(%rip), %ecx  # 2380
# DISASM-PIC-NEXT:             cmpl  {{.*}}(%rip), %edx  # 2380
# DISASM-PIC-NEXT:             orl   {{.*}}(%rip), %edi  # 2380
# DISASM-PIC-NEXT:             sbbl  {{.*}}(%rip), %esi  # 2380
# DISASM-PIC-NEXT:             subl  {{.*}}(%rip), %ebp  # 2380
# DISASM-PIC-NEXT:             xorl  {{.*}}(%rip), %r8d  # 2380
# DISASM-PIC-NEXT:             testl %r15d, {{.*}}(%rip) # 2380
# DISASM-PIC-NEXT: 12a0:       adcq  {{.*}}(%rip), %rax  # 2380
# DISASM-PIC-NEXT:             addq  {{.*}}(%rip), %rbx  # 2380
# DISASM-PIC-NEXT:             andq  {{.*}}(%rip), %rcx  # 2380
# DISASM-PIC-NEXT:             cmpq  {{.*}}(%rip), %rdx  # 2380
# DISASM-PIC-NEXT:             orq   {{.*}}(%rip), %rdi  # 2380
# DISASM-PIC-NEXT:             sbbq  {{.*}}(%rip), %rsi  # 2380
# DISASM-PIC-NEXT:             subq  {{.*}}(%rip), %rbp  # 2380
# DISASM-PIC-NEXT:             xorq  {{.*}}(%rip), %r8   # 2380
# DISASM-PIC-NEXT:             testq %r15, {{.*}}(%rip)  # 2380

.data
.type   bar, @object
bar:
 .byte   1
 .size   bar, .-bar

.text
.globl  _start
.type   _start, @function
_start:
## R_X86_64_GOTPCRELX
  adcl    bar@GOTPCREL(%rip), %eax
  addl    bar@GOTPCREL(%rip), %ebx
  andl    bar@GOTPCREL(%rip), %ecx
  cmpl    bar@GOTPCREL(%rip), %edx
  orl     bar@GOTPCREL(%rip), %edi
  sbbl    bar@GOTPCREL(%rip), %esi
  subl    bar@GOTPCREL(%rip), %ebp
  xorl    bar@GOTPCREL(%rip), %r8d
  testl   %r15d, bar@GOTPCREL(%rip)

## R_X86_64_REX_GOTPCRELX
  adcq    bar@GOTPCREL(%rip), %rax
  addq    bar@GOTPCREL(%rip), %rbx
  andq    bar@GOTPCREL(%rip), %rcx
  cmpq    bar@GOTPCREL(%rip), %rdx
  orq     bar@GOTPCREL(%rip), %rdi
  sbbq    bar@GOTPCREL(%rip), %rsi
  subq    bar@GOTPCREL(%rip), %rbp
  xorq    bar@GOTPCREL(%rip), %r8
  testq   %r15, bar@GOTPCREL(%rip)
