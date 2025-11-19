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

## -z dynamic-undefined-weak is ignored if .dynsym is absent (-no-pie without DSO)
# RUN: ld.lld a.o -o a.d -z dynamic-undefined-weak 2>&1 | count 0
# RUN: llvm-readelf -r --hex-dump=.data a.d | FileCheck %s --check-prefix=STATIC

## Currently no effect for S+A relocations.
# RUN: ld.lld a.o s.so -o as.d -z dynamic-undefined-weak
# RUN: llvm-readelf -r --hex-dump=.data as.d | FileCheck %s --check-prefix=STATIC

## -z dynamic-undefined-weak forces dynamic relocations if .dynsym is present.
# RUN: ld.lld a.o -o a.pie.d -pie -z dynamic-undefined-weak
# RUN: llvm-readelf -r a.pie.d | FileCheck %s --check-prefix=DYN

## -z nodynamic-undefined-weak suppresses dynamic relocations.
# RUN: ld.lld a.o -o a.so.n -shared -z dynamic-undefined-weak -z nodynamic-undefined-weak
# RUN: llvm-readelf -r --hex-dump=.data a.so.n | FileCheck %s --check-prefix=STATIC

# STATIC:      no relocations
# STATIC:      Hex dump of section '.data':
# STATIC-NEXT: {{.*}} 00000000 00000000 03000000 00000000 .
# STATIC-EMPTY:

# DYN:        Relocation section '.rela.dyn' {{.*}} contains 2
# DYN:        R_X86_64_64 0000000000000000 foobar + 0{{$}}

# RUN: ld.lld a.o b.o -o ab -z undefs
# RUN: llvm-readelf -r -x .data ab | FileCheck %s --check-prefix=STATIC1
# RUN: ld.lld a.o b.o s.so -o abs -z undefs
# RUN: llvm-readelf -r -x .data abs | FileCheck %s --check-prefix=STATIC1
# RUN: ld.lld a.o b.o -o ab.pie -pie -z undefs
# RUN: llvm-readelf -r -x .data ab.pie | FileCheck %s --check-prefix=STATIC1
# RUN: ld.lld a.o b.o s.so -o abs.pie -pie -z undefs
# RUN: llvm-readelf -r -x .data abs.pie | FileCheck %s --check-prefix=DYN1

# STATIC1:      no relocations
# STATIC1:      Hex dump of section '.data':
# STATIC1-NEXT: {{.*}} 00000000 00000000 03000000 00000000 .
# STATIC1-NEXT: {{.*}} 05000000 00000000                   .
# STATIC1-EMPTY:

# DYN1:        Relocation section '.rela.dyn' {{.*}} contains 3
# DYN1:        Hex dump of section '.data':
# DYN1-NEXT:   {{.*}} 00000000 00000000 00000000 00000000 .
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
