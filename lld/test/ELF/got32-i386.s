# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex --section-headers -d %t | FileCheck %s

## We have R_386_GOT32 relocation here.
.globl foo
.type foo, @function
foo:
 nop

_start:
 movl foo@GOT, %ebx

## 73728 == 0x12000 == ADDR(.got)
# CHECK: Sections:
# CHECK:  Name Size     VMA
# CHECK:  .got 00000004 004020fc
# CHECK:       <_start>:
# CHECK-NEXT:   4010f5: 8b 1d {{.*}}  movl 4202748, %ebx

# RUN: not ld.lld %t.o -o /dev/null -pie 2>&1 | FileCheck %s --check-prefix=ERR
# ERR: error: relocation R_386_GOT32 cannot be used against symbol 'foo'; recompile with -fPIC
