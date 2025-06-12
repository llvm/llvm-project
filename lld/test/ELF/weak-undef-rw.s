# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o s.o
# RUN: ld.lld -shared s.o -o s.so

# RUN: ld.lld a.o -o a --export-dynamic
# RUN: llvm-readelf -r --hex-dump=.data a | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld a.o s.so -o as
# RUN: llvm-readelf -r --hex-dump=.data as | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld a.o -o a.pie -pie
# RUN: llvm-readelf -r --hex-dump=.data a.pie | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld a.o -o a.so -shared
# RUN: llvm-readelf -r a.so | FileCheck %s --check-prefix=DYN

## gABI leaves the behavior of weak undefined references implementation defined.
## We choose to resolve them statically for static linking and produce dynamic relocations
## for dynamic linking (-shared or at least one input DSO).
##
## Note: Some ports of GNU ld support -z nodynamic-undefined-weak that we don't
## implement.

# STATIC:      no relocations
# STATIC:      Hex dump of section '.data':
# STATIC-NEXT: {{.*}} 00000000 00000000 03000000 00000000 .
# STATIC-EMPTY:

# DYN:        Relocation section '.rela.dyn' {{.*}} contains 2
# DYN:        R_X86_64_64 0000000000000000 foobar + 0{{$}}

# RUN: ld.lld a.o b.o -o ab -z undefs
# RUN: llvm-readelf -r -x .data ab | FileCheck %s --check-prefix=STATIC1
# RUN: ld.lld a.o b.o s.so -o abs -z undefs
# RUN: llvm-readelf -r -x .data abs | FileCheck %s --check-prefix=DYN1
# RUN: ld.lld a.o b.o -o abs.pie -pie -z undefs
# RUN: llvm-readelf -r -x .data abs.pie | FileCheck %s --check-prefix=STATIC1

# STATIC1:      no relocations
# STATIC1:      Hex dump of section '.data':
# STATIC1-NEXT: {{.*}} 00000000 00000000 03000000 00000000 .
# STATIC1-NEXT: {{.*}} 05000000 00000000                   .
# STATIC1-EMPTY:

# DYN1:        Relocation section '.rela.dyn' {{.*}} contains 1
# DYN1:        Hex dump of section '.data':
# DYN1-NEXT:   {{.*}} 00000000 00000000 03000000 00000000 .
# DYN1-NEXT:   {{.*}} 00000000 00000000                   .
# DYN1-EMPTY:

# RUN: ld.lld a.o b.o c.o -pie -z undefs 2>&1 | count 0

#--- a.s
.global _start
_start:
.data
.weak foobar
.quad foobar
.quad foobar+3

#--- b.s
.data
.quad undef+5

#--- c.s
call undef
