// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/map-file2.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/map-file3.s -o %t3.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/map-file4.s -o %t4.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/map-file5.s -o %t5.o
// RUN: ld.lld -shared %t5.o -o %t5.so -soname dso
// RUN: rm -f %t4.a
// RUN: llvm-ar rc %t4.a %t4.o
// RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t -M | FileCheck -strict-whitespace %s
// RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t -print-map | FileCheck -strict-whitespace %s
// RUN: ld.lld %t1.o %t2.o %t3.o %t4.a %t5.so -o %t -Map=%t.map
// RUN: FileCheck -strict-whitespace %s < %t.map

.global _start
_start:
 .quad sharedFoo
 .quad sharedBar
        call baz
.global _Z1fi
_Z1fi:
.cfi_startproc
.cfi_endproc
nop
.weak bar
bar:
.long bar - .
.long zed - .
local:
.comm   common,4,16
.global abs
abs = 0xAB5
labs = 0x1AB5

// CHECK:      Address          Size             Align Out     In      Symbol
// CHECK-NEXT: 00000000002001c8 0000000000000048     8 .dynsym
// CHECK-NEXT: 00000000002001c8 0000000000000048     8         <internal>:(.dynsym)
// CHECK-NEXT: 0000000000200210 0000000000000024     8 .gnu.hash
// CHECK-NEXT: 0000000000200210 0000000000000024     8         <internal>:(.gnu.hash)
// CHECK-NEXT: 0000000000200234 0000000000000020     4 .hash
// CHECK-NEXT: 0000000000200234 0000000000000020     4         <internal>:(.hash)
// CHECK-NEXT: 0000000000200254 0000000000000019     1 .dynstr
// CHECK-NEXT: 0000000000200254 0000000000000019     1         <internal>:(.dynstr)
// CHECK-NEXT: 0000000000200270 0000000000000030     8 .rela.dyn
// CHECK-NEXT: 0000000000200270 0000000000000030     8         <internal>:(.rela.dyn)
// CHECK-NEXT: 00000000002002a0 0000000000000030     8 .eh_frame
// CHECK-NEXT: 00000000002002a0 0000000000000030     8         <internal>:(.eh_frame)
// CHECK-NEXT: 0000000000201000 0000000000000025     4 .text
// CHECK-NEXT: 0000000000201000 000000000000001e     4         {{.*}}{{/|\\}}map-file.s.tmp1.o:(.text)
// CHECK-NEXT: 0000000000201000 0000000000000000     0                 _start
// CHECK-NEXT: 0000000000201015 0000000000000000     0                 f(int)
// CHECK-NEXT: 000000000020101e 0000000000000000     0                 local
// CHECK-NEXT: 0000000000201020 0000000000000002     4         {{.*}}{{/|\\}}map-file.s.tmp2.o:(.text)
// CHECK-NEXT: 0000000000201020 0000000000000000     0                 foo
// CHECK-NEXT: 0000000000201021 0000000000000000     0                 bar
// CHECK-NEXT: 0000000000201022 0000000000000000     1         {{.*}}{{/|\\}}map-file.s.tmp2.o:(.text.zed)
// CHECK-NEXT: 0000000000201022 0000000000000000     0                 zed
// CHECK-NEXT: 0000000000201024 0000000000000000     4         {{.*}}{{/|\\}}map-file.s.tmp3.o:(.text)
// CHECK-NEXT: 0000000000201024 0000000000000000     0                 bah
// CHECK-NEXT: 0000000000201024 0000000000000001     4         {{.*}}{{/|\\}}map-file.s.tmp4.a(map-file.s.tmp4.o):(.text)
// CHECK-NEXT: 0000000000201024 0000000000000000     0                 baz
// CHECK-NEXT: 0000000000202000 00000000000000c0     8 .dynamic
// CHECK-NEXT: 0000000000202000 00000000000000c0     8         <internal>:(.dynamic)
// CHECK-NEXT: 0000000000203000 0000000000000010    16 .bss
// CHECK-NEXT: 0000000000203000 0000000000000004    16         {{.*}}{{/|\\}}map-file.s.tmp1.o:(COMMON)
// CHECK-NEXT: 0000000000203000 0000000000000004     0                 common
// CHECK-NEXT: 0000000000203004 0000000000000004     1         <internal>:(.bss)
// CHECK-NEXT: 0000000000203004 0000000000000004     0                 sharedFoo
// CHECK-NEXT: 0000000000203008 0000000000000008     1         <internal>:(.bss)
// CHECK-NEXT: 0000000000203008 0000000000000008     0                 sharedBar
// CHECK-NEXT: 0000000000000000 0000000000000008     1 .comment
// CHECK-NEXT: 0000000000000000 0000000000000008     1         <internal>:(.comment)
// CHECK-NEXT: 0000000000000000 0000000000000168     8 .symtab
// CHECK-NEXT: 0000000000000000 0000000000000168     8         <internal>:(.symtab)
// CHECK-NEXT: 0000000000000000 000000000000006c     1 .shstrtab
// CHECK-NEXT: 0000000000000000 000000000000006c     1         <internal>:(.shstrtab)
// CHECK-NEXT: 0000000000000000 0000000000000055     1 .strtab
// CHECK-NEXT: 0000000000000000 0000000000000055     1         <internal>:(.strtab)

// RUN: not ld.lld %t1.o %t2.o %t3.o %t4.a -o %t -Map=/ 2>&1 \
// RUN:  | FileCheck -check-prefix=FAIL %s
// FAIL: cannot open map file /
