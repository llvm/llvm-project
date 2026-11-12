# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld -z norelro %t.o -o %t
# RUN: llvm-readelf -x .init -x .fini -x .init_array -x .fini_array %t | \
# RUN:   FileCheck --check-prefixes=CHECK,ORDERED %s

# RUN: ld.lld -z norelro %t.o --shuffle-sections '*=1' -o %t1
# RUN: llvm-readelf -x .init -x .fini -x .init_array -x .fini_array %t1 | \
# RUN:   FileCheck --check-prefixes=CHECK,SHUFFLED %s

## .init and .fini rely on a particular order, e.g. crti.o crtbegin.o crtend.o crtn.o
## Don't shuffle them.
# CHECK:      Hex dump of section '.init'
# CHECK-NEXT: 00010203 04050607 08090a0b

# CHECK:      Hex dump of section '.fini'
# CHECK-NEXT: 00010203 04050607 08090a0b

## SHT_INIT_ARRAY/SHT_FINI_ARRAY with explicit priorities are still ordered.
# CHECK:      Hex dump of section '.init_array'
# CHECK-NEXT: 0x{{[0-9a-f]+}} ff
# ORDERED-SAME: 000102 03040506 0708090a 0b
# SHUFFLED-SAME: 070201 0006090a 040b0503 08

# CHECK:      Hex dump of section '.fini_array'
# CHECK-NEXT: 0x{{[0-9a-f]+}} ff
# ORDERED-SAME:  000102 03040506 0708090a 0b
# SHUFFLED-SAME: 070008 0a040209 03010b06 05

## With a SECTIONS command, SHT_INIT_ARRAY prirotities are ignored.
## All .init_array* are shuffled together.
# RUN: echo 'SECTIONS { \
# RUN:   .init_array : { *(.init_array*) } \
# RUN:   .fini_array : { *(.fini_array*) }}' > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t2
# RUN: llvm-readelf -x .init -x .fini -x .init_array -x .fini_array %t2 | \
# RUN:   FileCheck --check-prefixes=CHECK2,ORDERED2 %s
# RUN: ld.lld -T %t.script %t.o --shuffle-sections '*=1' -o %t3
# RUN: llvm-readelf -x .init -x .fini -x .init_array -x .fini_array %t3 | \
# RUN:   FileCheck --check-prefixes=CHECK2,SHUFFLED2 %s

# CHECK2:       Hex dump of section '.init_array'
# ORDERED2-NEXT:  0x{{[0-9a-f]+}} 00010203 04050607 08090a0b ff
# SHUFFLED2-NEXT: 0x{{[0-9a-f]+}} 07020100 06090a04 0b050308 ff

.irp i,0,1,2,3,4,5,6,7,8,9,10,11
  .section .init,"ax",@progbits,unique,\i
  .byte \i
  .section .fini,"ax",@progbits,unique,\i
  .byte \i
  .section .init_array,"aw",@init_array,unique,\i
  .byte \i
  .section .fini_array,"aw",@fini_array,unique,\i
  .byte \i
.endr

.section .init_array.1,"aw",@init_array
.byte 255
.section .fini_array.1,"aw",@fini_array
.byte 255
