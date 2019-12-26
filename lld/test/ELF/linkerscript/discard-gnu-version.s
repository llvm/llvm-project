# REQUIRES: x86
# RUN: echo '.globl f; f:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: echo 'v1 { f; };' > %t1.ver
# RUN: ld.lld -shared --version-script %t1.ver %t1.o -o %t1.so

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'v1 { foo; };' > %t.ver
# RUN: ld.lld -shared --version-script %t.ver %t.o %t1.so -o %t.so
# RUN: llvm-readelf -S -d %t.so | FileCheck --check-prefixes=SYM,DEF,NEED %s

# SYM:  .gnu.version
# DEF:  .gnu.version_d
# NEED: .gnu.version_r

# SYM:  (VERSYM)
# DEF:  (VERDEF)
# DEF:  (VERDEFNUM)
# NEED: (VERNEED)
# NEED: (VERNEEDNUM)

## Discard all of .gnu.version*
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.gnu.version*) } }' > %t.script
# RUN: ld.lld -shared --version-script %t.ver -T %t.script %t.o %t1.so -o %t.so
# RUN: llvm-readelf -S -d %t.so | FileCheck /dev/null \
# RUN:   --implicit-check-not='(VER' --implicit-check-not=.gnu.version

## Discard .gnu.version
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.gnu.version) } }' > %t.noversym.script
# RUN: ld.lld -shared --version-script %t.ver -T %t.noversym.script %t.o %t1.so -o %t.noversym.so
# RUN: llvm-readelf -S -d %t.noversym.so | FileCheck --check-prefixes=DEF,NEED %s \
# RUN:   --implicit-check-not='(VER' --implicit-check-not=.gnu.version

## Discard .gnu.version_d
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.gnu.version_d) } }' > %t.noverdef.script
# RUN: ld.lld -shared --version-script %t.ver -T %t.noverdef.script %t.o %t1.so -o %t.noverdef.so
# RUN: llvm-readelf -S -d %t.noverdef.so | FileCheck --check-prefixes=SYM,NEED %s \
# RUN:   --implicit-check-not='(VER' --implicit-check-not=.gnu.version

## Discard .gnu.version_r
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.gnu.version_r) } }' > %t.noverneed.script
# RUN: ld.lld -shared --version-script %t.ver -T %t.noverneed.script %t.o %t1.so -o %t.noverneed.so
# RUN: llvm-readelf -S -d %t.noverneed.so | FileCheck --check-prefixes=SYM,DEF %s \
# RUN:   --implicit-check-not='(VER' --implicit-check-not=.gnu.version

.globl foo
foo:
  call f@plt
