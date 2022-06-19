# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/defs.s -o %t/defs.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs1.s -o %t/refs1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs2.s -o %t/refs2.o
# RUN: %lld -lSystem -dylib %t/defs.o -o %t/libdefs.dylib
# RUN: %lld -lSystem -dylib --icf=all %t/refs1.o %t/refs2.o %t/libdefs.dylib -o %t/out
# RUN: llvm-objdump --macho --section-headers --bind %t/out | FileCheck %s \
# RUN:   --implicit-check-not __objc_classrefs

## Check that we only have 3 (unique) entries
# CHECK:      Sections:
# CHECK-NEXT: Idx Name             Size
# CHECK:          __objc_classrefs 00000018

## And only two binds
# CHECK:       Bind table:
# CHECK-NEXT:  segment  section           address  type     addend dylib    symbol
# CHECK-DAG:   __DATA   __objc_classrefs  {{.*}}   pointer       0 libdefs  _OBJC_CLASS_$_Bar
# CHECK-DAG:   __DATA   __objc_classrefs  {{.*}}   pointer       0 libdefs  _OBJC_CLASS_$_Foo

#--- defs.s
.globl _OBJC_CLASS_$_Foo, _OBJC_CLASS_$_Bar
.section __DATA,__objc_data
_OBJC_CLASS_$_Foo:
 .quad 123

_OBJC_CLASS_$_Bar:
 .quad 456

.subsections_via_symbols

#--- refs1.s
.globl _OBJC_CLASS_$_Baz

.section __DATA,__objc_data
_OBJC_CLASS_$_Baz:
 .quad 789

.section __DATA,__objc_classrefs
.quad _OBJC_CLASS_$_Foo
.quad _OBJC_CLASS_$_Bar
.quad _OBJC_CLASS_$_Baz
.quad _OBJC_CLASS_$_Baz

.subsections_via_symbols

#--- refs2.s
.section __DATA,__objc_classrefs
.quad _OBJC_CLASS_$_Foo
.quad _OBJC_CLASS_$_Bar

.subsections_via_symbols
