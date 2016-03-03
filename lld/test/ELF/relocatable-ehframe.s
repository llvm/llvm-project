# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/relocatable-ehframe.s -o %t2.o
# RUN: ld.lld -r %t1.o %t2.o -o %t
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols -r %t | FileCheck %s
# RUN: llvm-objdump -s -d %t | FileCheck -check-prefix=CHECKTEXT %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.eh_frame {
# CHECK-NEXT:     0x20 R_X86_64_PC32 foo 0x0
# CHECK-NEXT:     0x34 R_X86_64_PC32 bar 0x0
# CHECK-NEXT:     0x48 R_X86_64_PC32 dah 0x0
# CHECK-NEXT:     0x78 R_X86_64_PC32 foo1 0x0
# CHECK-NEXT:     0x8C R_X86_64_PC32 bar1 0x0
# CHECK-NEXT:     0xA0 R_X86_64_PC32 dah1 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECKTEXT:      Contents of section .strtab:
# CHECKTEXT-NEXT:  0000 00666f6f 00626172 00646168 00666f6f  .foo.bar.dah.foo
# CHECKTEXT-NEXT:  0010 31006261 72310064 61683100 5f737461  1.bar1.dah1._sta
# CHECKTEXT-NEXT:  0020 727400                               rt.
 
.section foo,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.section bar,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.section dah,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.text
.globl _start;
_start:
 nop
