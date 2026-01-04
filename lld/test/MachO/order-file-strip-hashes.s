# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/a.out %t/a.o -order_file %t/ord-1
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/a.out | FileCheck %s
	
#--- a.s
.text
.globl _main, A, _B, C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222

_main:
  ret
A:
  ret
F:
  add w0, w0, #3
  bl C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222
  ret
C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222:
  add w0, w0, #2
  bl  A
  ret
D:
  add w0, w0, #2
  bl B
  ret
B:
  add w0, w0, #1
  bl  A
  ret
E:
  add w0, w0, #2
  bl C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222
  ret

.section __DATA,__objc_const
# test multiple symbols at the same address, which will be alphabetic sorted based symbol names
_OBJC_$_CATEGORY_CLASS_METHODS_Foo_$_Cat2:
  .quad 789

_OBJC_$_CATEGORY_SOME_$_FOLDED:
_OBJC_$_CATEGORY_Foo_$_Cat1:
_ALPHABETIC_SORT_FIRST:
 .quad 123

_OBJC_$_CATEGORY_Foo_$_Cat2:
 .quad 222

_OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat1:
  .quad 456

.section __DATA,__objc_data
_OBJC_CLASS_$_Foo:
 .quad 123

_OBJC_CLASS_$_Bar.llvm.1234:
 .quad 456

_OBJC_CLASS_$_Baz:
 .quad 789

_OBJC_CLASS_$_Baz2:
 .quad 999

.section __DATA,__objc_classrefs
.quad _OBJC_CLASS_$_Foo
.quad _OBJC_CLASS_$_Bar.llvm.1234
.quad _OBJC_CLASS_$_Baz

.subsections_via_symbols


#--- ord-1
# change order, parital covered
A
B
C.__uniq.555555555555555555555555555555555555555.llvm.6666666666666666666
_OBJC_CLASS_$_Baz
_OBJC_CLASS_$_Bar.__uniq.12345
_OBJC_CLASS_$_Foo.__uniq.123.llvm.123456789
_OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat1
_OBJC_$_CATEGORY_Foo_$_Cat1.llvm.1234567

# .text
# CHECK: A
# CHECK: B
# CHECK: C
# .section __DATA,__objc_const
# CHECK: _OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat1
# CHECK: _OBJC_$_CATEGORY_Foo_$_Cat1
# .section __DATA,__objc_data
# CHECK: _OBJC_CLASS_$_Baz
# CHECK: _OBJC_CLASS_$_Bar
# CHECK: _OBJC_CLASS_$_Foo
