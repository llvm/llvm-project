# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

#--- 1.lds
PHDRS {all PT_LOAD FILEHDR PHDRS ;}
SECTIONS {
  . = 0x10000200;
  .text : {*(.text*)} :all
  .foo : {*(.foo.*)} :"all"
  .data : {*(.data.*)} : "all"}

# RUN: ld.lld -o 1 -T 1.lds a.o
# RUN: llvm-readelf -Sl 1 | FileCheck %s
# CHECK:      [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      [ 1] .text             PROGBITS        0000000010000200 000200 000001 00  AX  0   0  4
# CHECK-NEXT: [ 2] .foo              PROGBITS        0000000010000201 000201 000008 00  WA  0   0  1

# CHECK:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD           0x000000 0x0000000010000000 0x0000000010000000 0x000209 0x000209 RWE 0x1000

#--- 2.lds
## Check that program headers are not written, unless we explicitly tell
## lld to do this.
PHDRS {all PT_LOAD;}
SECTIONS {
    . = 0x10000200;
    /DISCARD/ : {*(.text*)}
    .foo : {*(.foo.*)} :all
}

# RUN: ld.lld -o 2 -T 2.lds a.o
# RUN: llvm-readelf -l 2 | FileCheck --check-prefix=NOPHDR %s
# NOPHDR:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# NOPHDR-NEXT: LOAD           0x000200 0x0000000010000200 0x0000000010000200 0x000008 0x000008 RW  0x1000

#--- 3.lds
PHDRS {all PT_LOAD FILEHDR PHDRS ;}
SECTIONS {
    . = 0x10000200;
    .text : {*(.text*)} :all
    .foo : {*(.foo.*)}
    .data : {*(.data.*)} }

# RUN: ld.lld -o 3 -T 3.lds a.o
# RUN: llvm-readelf -l 3 | FileCheck --check-prefix=DEFHDR %s
# DEFHDR:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# DEFHDR-NEXT: LOAD           0x000000 0x0000000010000000 0x0000000010000000 0x000209 0x000209 RWE 0x1000

#--- at.lds
## Check the AT(expr)
PHDRS {all PT_LOAD FILEHDR PHDRS AT(0x500 + 0x500) ;}
SECTIONS {
    . = 0x10000200;
    .text : {*(.text*)} :all
    .foo : {*(.foo.*)} :all
    .data : {*(.data.*)} :all}

# RUN: ld.lld -o at -T at.lds a.o
# RUN: llvm-readelf -l at | FileCheck --check-prefix=AT %s
# AT:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# AT-NEXT: LOAD           0x000000 0x0000000010000000 0x0000000000000a00 0x000209 0x000209 RWE 0x1000

#--- int.lds
## Check the numetic values for PHDRS.
PHDRS {text PT_LOAD FILEHDR PHDRS; foo 0x11223344; }
SECTIONS { . = SIZEOF_HEADERS; .foo : { *(.foo* .text*) } : text : foo}

# RUN: ld.lld -o int -T int.lds a.o
# RUN: llvm-readelf -l int | FileCheck --check-prefix=INT-PHDRS %s
# INT-PHDRS:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# INT-PHDRS-NEXT: LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x0000b9 0x0000b9 RWE 0x1000
# INT-PHDRS-NEXT: <unknown>: 0x11223344 0x0000b0 0x00000000000000b0 0x00000000000000b0 0x000009 0x000009 RWE 0x4

#--- unspecified.lds
## Check that error is reported when trying to use phdr which is not listed
## inside PHDRS {} block
## TODO: If script doesn't contain PHDRS {} block then default phdr is always
## created and error is not reported.
PHDRS { all PT_LOAD; }
SECTIONS { .baz : {*(.foo.*)} :bar }

# RUN: not ld.lld -T unspecified.lds a.o 2>&1 | FileCheck --check-prefix=UNSPECIFIED %s
# UNSPECIFIED: unspecified.lds:6: program header 'bar' is not listed in PHDRS

#--- foohdr.lds
PHDRS { text PT_LOAD FOOHDR; }

# RUN: not ld.lld -T foohdr.lds a.o 2>&1 | FileCheck --check-prefix=FOOHDR %s
# FOOHDR: error: foohdr.lds:1: unexpected header attribute: FOOHDR

#--- pt_foo.lds
PHDRS { text PT_FOO FOOHDR; }

# RUN: not ld.lld -T pt_foo.lds a.o 2>&1 | FileCheck --check-prefix=PTFOO %s --strict-whitespace
#      PTFOO:{{.*}}error: pt_foo.lds:1: invalid program header type: PT_FOO
# PTFOO-NEXT:>>> PHDRS { text PT_FOO FOOHDR; }
# PTFOO-NEXT:>>>              ^

#--- unclosed.lds
PHDRS { text PT_LOAD ;

# RUN: not ld.lld -T unclosed.lds a.o 2>&1 | FileCheck --check-prefix=UNCLOSED %s
#     UNCLOSED:error: unclosed.lds:1: unexpected EOF
# UNCLOSED-NOT:{{.}}

#--- unclosed2.lds
PHDRS { text PT_LOAD

# RUN: not ld.lld -T unclosed2.lds a.o 2>&1 | FileCheck --check-prefix=UNCLOSED2 %s
# UNCLOSED2: error: unclosed2.lds:1: unexpected header attribute:

#--- a.s
.global _start
_start:
 nop

.section .foo.1,"a"
foo1:
 .long 0

.section .foo.2,"aw"
foo2:
 .long 0
