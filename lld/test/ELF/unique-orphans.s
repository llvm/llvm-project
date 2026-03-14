# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

.section .foo,"a",@progbits,unique,1
.byte 1

.section .foo,"a",@progbits,unique,2
.byte 2

.section .foo,"a",@progbits,unique,3
.byte 3

## We should have 3 instances of orphan section foo.
## Test with -r
# RUN: ld.lld %t.o -o %t.elf --unique 
# RUN: llvm-readelf -S %t.elf | FileCheck %s

# CHECK-COUNT-3: .foo
# CHECK-NOT: .foo

## Test that --unique does not affect sections specified in output section descriptions.
# RUN: echo 'SECTIONS { .foo : { *(.foo) }}' > %t.script
# RUN: ld.lld %t.o -o %t2.elf -T %t.script --unique 
# RUN: llvm-readelf -S %t2.elf | FileCheck --check-prefix SCRIPT %s
# SCRIPT: .foo
# SCRIPT-NOT: .foo
