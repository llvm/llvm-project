# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: not ld.lld a.o -T 1.lds 2>&1 | FileCheck %s --check-prefix=ERR1
# ERR1: error: 1.lds:1: there is a cycle in linker script INCLUDEs

# RUN: not ld.lld a.o -T 2a.lds 2>&1 | FileCheck %s --check-prefix=ERR2
# ERR2: error: 2b.lds:1: there is a cycle in linker script INCLUDEs

# RUN: ld.lld a.o -T 3.lds -o 3
# RUN: llvm-objdump -s 3 | FileCheck %s --check-prefix=CHECK3
# CHECK3:      Contents of section foo:
# CHECK3-NEXT: 0000 2a2a                    **

#--- 0.lds
BYTE(42)
#--- 1.lds
INCLUDE "1.lds"
#--- 2a.lds
INCLUDE "2b.lds"
#--- 2b.lds
INCLUDE "2a.lds"
#--- 3.lds
SECTIONS {
  foo : { INCLUDE "0.lds" INCLUDE "0.lds" }
}

#--- a.s
.globl _start
_start:
  ret
