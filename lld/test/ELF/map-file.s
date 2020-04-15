# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/map-file2.s -o %t2.o
# RUN: echo '.global bah; bah:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t3.o
# RUN: echo '.global baz; baz: ret' | llvm-mc -filetype=obj -triple=x86_64 - -o %t4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/map-file5.s -o %t5.o
# RUN: ld.lld -shared %t5.o -o %t5.so -soname dso
# RUN: rm -f %t4.a
# RUN: llvm-ar rc %t4.a %t4.o
# RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t -M | FileCheck --match-full-lines --strict-whitespace %s
# RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t --print-map | FileCheck --match-full-lines -strict-whitespace %s
# RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t -Map=%t.map
# RUN: FileCheck -strict-whitespace %s < %t.map

.global _start
_start:
.cfi_startproc
.cfi_endproc
 .quad sharedFoo
 .quad sharedBar
 .byte 0xe8
 .long sharedFunc1 - .
 .byte 0xe8
 .long sharedFunc2 - .
 .byte 0xe8
 .long baz - .
.global _Z1fi
_Z1fi:
.cfi_startproc
nop
.cfi_endproc
.weak bar
bar:
.long bar - .
.long zed - .
local:
.comm   common,4,16
.global abs
abs = 0xAB5
labs = 0x1AB5

##           0123456789abcdef 0123456789abcdef
#      CHECK:             VMA              LMA     Size Align Out     In      Symbol
# CHECK-NEXT:          200200           200200       78     8 .dynsym
# CHECK-NEXT:          200200           200200       78     8         <internal>:(.dynsym)
# CHECK-NEXT:          200278           200278       2c     8 .gnu.hash
# CHECK-NEXT:          200278           200278       2c     8         <internal>:(.gnu.hash)
# CHECK-NEXT:          2002a4           2002a4       30     4 .hash
# CHECK-NEXT:          2002a4           2002a4       30     4         <internal>:(.hash)
# CHECK-NEXT:          2002d4           2002d4       31     1 .dynstr
# CHECK-NEXT:          2002d4           2002d4       31     1         <internal>:(.dynstr)
# CHECK-NEXT:          200308           200308       30     8 .rela.dyn
# CHECK-NEXT:          200308           200308       30     8         <internal>:(.rela.dyn)
# CHECK-NEXT:          200338           200338       30     8 .rela.plt
# CHECK-NEXT:          200338           200338       30     8         <internal>:(.rela.plt)
# CHECK-NEXT:          200368           200368       64     8 .eh_frame
# CHECK-NEXT:          200368           200368       2c     1         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.eh_frame+0x0)
# CHECK-NEXT:          200398           200398       14     1         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.eh_frame+0x2c)
# CHECK-NEXT:          2003b0           2003b0       18     1         {{.*}}{{/|\\}}map-file.s.tmp2.o:(.eh_frame+0x18)
# CHECK-NEXT:          2013cc           2013cc       2d     4 .text
# CHECK-NEXT:          2013cc           2013cc       28     4         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.text)
# CHECK-NEXT:          2013cc           2013cc        0     1                 _start
# CHECK-NEXT:          2013eb           2013eb        0     1                 f(int)
# CHECK-NEXT:          2013f4           2013f4        0     1                 local
# CHECK-NEXT:          2013f4           2013f4        2     4         {{.*}}{{/|\\}}map-file.s.tmp2.o:(.text)
# CHECK-NEXT:          2013f4           2013f4        0     1                 foo
# CHECK-NEXT:          2013f5           2013f5        0     1                 bar
# CHECK-NEXT:          2013f6           2013f6        0     1         {{.*}}{{/|\\}}map-file.s.tmp2.o:(.text.zed)
# CHECK-NEXT:          2013f6           2013f6        0     1                 zed
# CHECK-NEXT:          2013f8           2013f8        0     4         {{.*}}{{/|\\}}map-file.s.tmp3.o:(.text)
# CHECK-NEXT:          2013f8           2013f8        0     1                 bah
# CHECK-NEXT:          2013f8           2013f8        1     4         {{.*}}{{/|\\}}map-file.s.tmp4.a(map-file.s.tmp4.o):(.text)
# CHECK-NEXT:          2013f8           2013f8        0     1                 baz
# CHECK-NEXT:          201400           201400       30    16 .plt
# CHECK-NEXT:          201400           201400       30    16         <internal>:(.plt)
# CHECK-NEXT:          201410           201410        0     1                 sharedFunc1
# CHECK-NEXT:          201420           201420        0     1                 sharedFunc2
# CHECK-NEXT:          202430           202430      100     8 .dynamic
# CHECK-NEXT:          202430           202430      100     8         <internal>:(.dynamic)
# CHECK-NEXT:          203530           203530       28     8 .got.plt
# CHECk-NEXT:          203530           203530       28     8         <internal>:(.got.plt)
# CHECk-NEXT:          203560           203560       10    16 .bss
# CHECk-NEXT:          203560           203560        4    16         {{.*}}{{/|\\}}map-file.s.tmp1.o:(COMMON)
# CHECk-NEXT:          203560           203560        4     1                 common
# CHECk-NEXT:          203564           203564        4     1         <internal>:(.bss)
# CHECk-NEXT:          203564           203564        4     1                 sharedFoo
# CHECk-NEXT:          203568           203568        8     1         <internal>:(.bss)
# CHECk-NEXT:          203568           203568        8     1                 sharedBar
# CHECk-NEXT:               0                0       48     1 .comment
# CHECk-NEXT:               0                0       48     1         <internal>:(.comment)
# CHECk-NEXT:               0                0      198     8 .symtab
# CHECk-NEXT:               0                0      198     8         <internal>:(.symtab)
# CHECk-NEXT:               0                0       84     1 .shstrtab
# CHECk-NEXT:               0                0       84     1         <internal>:(.shstrtab)
# CHECk-NEXT:               0                0       6d     1 .strtab
# CHECk-NEXT:               0                0       6d     1         <internal>:(.strtab)

# RUN: not ld.lld %t1.o %t2.o %t3.o %t4.a -o /dev/null -Map=/ 2>&1 \
# RUN:  | FileCheck --check-prefix=FAIL %s
# FAIL: cannot open map file /
