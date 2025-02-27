# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --export-dynamic
# RUN: llvm-readelf -r --hex-dump=.data %t | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld %t.o -o %t.pie -pie
# RUN: llvm-readelf -r --hex-dump=.data %t.pie | FileCheck %s --check-prefix=STATIC
# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=PIC

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

        .global _start
_start:
        .data
        .weak foobar
        .quad foobar
