# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/bcde-abcd-abde.s -o %t/bcde-abcd-abde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/bbcde-abcdd.s -o %t/bbcde-abcdd.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/aabcde-abcdee.s -o %t/aabcde-abcdee.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/bacde.s -o %t/bacde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/__Z3fooPi.s -o %t/__Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/__Z3fooPKi-__Z3fooPi.s -o %t/__Z3fooPKi-__Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/__Z3FOOPKi.s -o %t/__Z3FOOPKi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/__Z3fooPKi-__Z3FOOPKi.s -o %t/__Z3fooPKi-__Z3FOOPKi.o

## Insert a character.
## The spell corrector is enabled for the first two "undefined symbol" diagnostics.
# RUN: not %lld %t/test.o %t/bcde-abcd-abde.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t/test.o

## Symbols defined in DSO can be suggested.
# RUN: %lld %t/test.o -dylib -o %t.dylib
# RUN: not %lld %t.dylib %t/bcde-abcd-abde.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t.dylib

# INSERT:      error: undefined symbol: abde
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NEXT: >>> did you mean: abcde
# INSERT-NEXT: >>> defined in: [[FILE]]
# INSERT:      error: undefined symbol: abcd
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NEXT: >>> did you mean: abcde
# INSERT-NEXT: >>> defined in: [[FILE]]
# INSERT:      error: undefined symbol: bcde
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NOT:  >>>

## Substitute a character.
# RUN: not %lld %t/test.o %t/bbcde-abcdd.o -o /dev/null 2>&1 | FileCheck --check-prefix=SUBST %s

# SUBST:      error: undefined symbol: abcdd
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde
# SUBST:      error: undefined symbol: bbcde
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde

## Delete a character.
# RUN: not %lld %t/test.o %t/aabcde-abcdee.o -o /dev/null 2>&1 | FileCheck --check-prefix=DELETE %s

# DELETE:      error: undefined symbol: abcdee
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde
# DELETE:      error: undefined symbol: aabcde
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde

## Transpose.
# RUN: not %lld %t/test.o %t/bacde.o -o /dev/null 2>&1 | FileCheck --check-prefix=TRANSPOSE %s

# TRANSPOSE:      error: undefined symbol: bacde
# TRANSPOSE-NEXT: >>> referenced by {{.*}}
# TRANSPOSE-NEXT: >>> did you mean: abcde

## Missing const qualifier.
# RUN: not %lld %t/test.o %t/__Z3fooPi.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s
## Local defined symbols.
# RUN: not %lld %t/__Z3fooPKi-__Z3fooPi.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s

# CONST:      error: undefined symbol: foo(int*)
# CONST-NEXT: >>> referenced by {{.*}}
# CONST-NEXT: >>> did you mean: foo(int const*)

## Case mismatch.
# RUN: not %lld %t/test.o %t/__Z3FOOPKi.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s
# RUN: not %lld %t/__Z3fooPKi-__Z3FOOPKi.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s

# CASE:      error: undefined symbol: FOO(int const*)
# CASE-NEXT: >>> referenced by {{.*}}
# CASE-NEXT: >>> did you mean: foo(int const*)

#--- test.s
.globl _main, abcde, __Z3fooPKi
_main:
abcde:
__Z3fooPKi:

#--- bcde-abcd-abde.s
call bcde
call abcd
call abde

# Creates a nullptr entry in ObjFile::symbols, to test we don't crash on that.
.section  __DWARF,__debug_aranges,regular,debug
ltmp1:
  .byte 0

.subsections_via_symbols

#--- bbcde-abcdd.s
call bbcde
call abcdd

#--- aabcde-abcdee.s
call aabcde
call abcdee

#--- bacde.s
call bacde

#--- __Z3fooPi.s
call __Z3fooPi

#--- __Z3fooPKi-__Z3fooPi.s
__Z3fooPKi: call __Z3fooPi

#--- __Z3FOOPKi.s
call __Z3FOOPKi

#--- __Z3fooPKi-__Z3FOOPKi.s
__Z3fooPKi: call __Z3FOOPKi
