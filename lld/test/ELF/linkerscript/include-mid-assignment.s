# REQUIRES: x86
## When recording `commandString`, an assignment in an INCLUDEd linker script
## that is missing the trailing ';' must not crash lld. Same for a PROVIDE
## whose inner buffer is exhausted before the closing ')'.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: not ld.lld a.o -T outer.lds 2>&1 | FileCheck %s
# RUN: not ld.lld a.o -T outer2.lds 2>&1 | FileCheck %s --check-prefix=CHECK2

# CHECK: error: outer.lds:2: ; expected, but got {{.*}}
# CHECK2: error: outer2.lds:2: ) expected, but got {{.*}}

#--- outer.lds
INCLUDE "inc.lds"
SECTIONS { .text : { *(.text*) } }

#--- inc.lds
foo = 1

#--- outer2.lds
INCLUDE "inc2.lds"
SECTIONS { .text : { *(.text*) } }

#--- inc2.lds
PROVIDE(bar = 1

#--- a.s
.globl _start
_start:
  ret
