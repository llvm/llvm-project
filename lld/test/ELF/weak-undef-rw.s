# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: ld.lld a.o -o nopie --export-dynamic
# RUN: llvm-readelf -r --hex-dump=.data nopie | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld a.o -o out.pie -pie
# RUN: llvm-readelf -r --hex-dump=.data out.pie | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld a.o -o out.so -shared
# RUN: llvm-readobj -r out.so | FileCheck %s --check-prefix=PIC

## gABI leaves the behavior of weak undefined references implementation defined.
## We choose to resolve them statically for static linking and produce dynamic relocations
## for dynamic linking (-shared or at least one input DSO).
##
## Note: Some ports of GNU ld support -z nodynamic-undefined-weak that we don't
## implement.

# STATIC:      no relocations
# STATIC:      Hex dump of section '.data':
# STATIC-NEXT: {{.*}} 00000000 00000000 .
# STATIC-EMPTY:

# PIC:      .rela.dyn {
# PIC-NEXT:   R_X86_64_64 foobar 0x0
# PIC-NEXT: }

# RUN: ld.lld a.o b.o -o out1 -z undefs
# RUN: llvm-readelf -r -x .data out1 | FileCheck %s --check-prefix=STATIC1
# RUN: ld.lld a.o b.o -o out1.pie -pie -z undefs
# RUN: llvm-readelf -r -x .data out1.pie | FileCheck %s --check-prefix=STATIC1

# STATIC1:      no relocations
# STATIC1:      Hex dump of section '.data':
# STATIC1-NEXT: {{.*}} 00000000 00000000 00000000 00000000 .
# STATIC1-EMPTY:

# RUN: ld.lld a.o b.o c.o -pie -z undefs 2>&1 | count 0

#--- a.s
        .global _start
_start:
        .data
        .weak foobar
        .quad foobar

#--- b.s
.data
.quad undef

#--- c.s
call undef
