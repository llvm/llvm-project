# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --export-dynamic
# RUN: llvm-readelf -r --hex-dump=.data %t | FileCheck %s --check-prefix=NOPIC
# RUN: ld.lld %t.o -o %t.pie -pie
# RUN: llvm-readobj -r %t.pie | FileCheck %s --check-prefix=PIC
# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=PIC

## gABI leaves the behavior of weak undefined references implementation defined.
## We choose to resolve them statically for -no-pie and produce dynamic relocations
## for -pie and -shared.
##
## Note: Some ports of GNU ld support -z nodynamic-undefined-weak that we don't
## implement.

# NOPIC: no relocations
# NOPIC: Hex dump of section '.data':
# NOPIC-NEXT: {{.*}} 00000000 00000000 
# NOPIC-EMPTY:

# PIC:      .rela.dyn {
# PIC-NEXT:   R_X86_64_64 foobar 0x0
# PIC-NEXT: }

        .global _start
_start:
        .data
        .weak foobar
        .quad foobar
