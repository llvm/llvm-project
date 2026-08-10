# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/defs.s -o %t/defs.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs1.s -o %t/refs1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs2.s -o %t/refs2.o
# RUN: %lld -lSystem -dylib %t/defs.o -o %t/libdefs.dylib
# RUN: %lld -lSystem -dylib --icf=all %t/refs1.o %t/refs2.o %t/libdefs.dylib -o %t/out
# RUN: llvm-otool -o %t/out | FileCheck %s

## Check that we only have 3 (unique) entries, of which two are bound at runtime
## (i.e. they are entries that have a static value of 0x0).
# CHECK:       Contents of (__DATA,__objc_classrefs) section
# CHECK-NEXT:  0000000000001008 0x0    _OBJC_CLASS_$_Foo
# CHECK-NEXT:  0000000000001010 0x0    _OBJC_CLASS_$_Bar
# CHECK-NEXT:  0000000000001018 0x1000 _OBJC_CLASS_$_Baz
# CHECK-EMPTY:

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
