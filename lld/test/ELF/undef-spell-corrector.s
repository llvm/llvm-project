# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/bcde-abcd-abde.s -o %t/bcde-abcd-abde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/bbcde-abcdd.s -o %t/bbcde-abcdd.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/aabcde-abcdee.s -o %t/aabcde-abcdee.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/bacde.s -o %t/bacde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/_Z3fooPi.s -o %t/_Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/_Z3fooPKi-_Z3fooPi.s -o %t/_Z3fooPKi-_Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/_Z3FOOPKi.s -o %t/_Z3FOOPKi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/_Z3fooPKi-_Z3FOOPKi.s -o %t/_Z3fooPKi-_Z3FOOPKi.o

## Insert a character.
## The spell corrector is enabled for the first two "undefined symbol" diagnostics.
# RUN: not ld.lld %t/test.o %t/bcde-abcd-abde.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t/test.o

## Symbols defined in DSO can be suggested.
# RUN: ld.lld %t/test.o -shared -o %t.so
# RUN: not ld.lld %t.so %t/bcde-abcd-abde.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t.so

# INSERT:      error: undefined symbol: bcde
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NEXT: >>> did you mean: abcde
# INSERT-NEXT: >>> defined in: [[FILE]]
# INSERT:      error: undefined symbol: abcd
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NEXT: >>> did you mean: abcde
# INSERT-NEXT: >>> defined in: [[FILE]]
# INSERT:      error: undefined symbol: abde
# INSERT-NEXT: >>> referenced by {{.*}}
# INSERT-NOT:  >>>

## Substitute a character.
# RUN: not ld.lld %t/test.o %t/bbcde-abcdd.o -o /dev/null 2>&1 | FileCheck --check-prefix=SUBST %s

# SUBST:      error: undefined symbol: bbcde
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde
# SUBST:      error: undefined symbol: abcdd
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde

## Delete a character.
# RUN: not ld.lld %t/test.o %t/aabcde-abcdee.o -o /dev/null 2>&1 | FileCheck --check-prefix=DELETE %s

# DELETE:      error: undefined symbol: aabcde
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde
# DELETE:      error: undefined symbol: abcdee
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde

## Transpose.
# RUN: not ld.lld %t/test.o %t/bacde.o -o /dev/null 2>&1 | FileCheck --check-prefix=TRANSPOSE %s

# TRANSPOSE:      error: undefined symbol: bacde
# TRANSPOSE-NEXT: >>> referenced by {{.*}}
# TRANSPOSE-NEXT: >>> did you mean: abcde

## Missing const qualifier.
# RUN: not ld.lld %t/test.o %t/_Z3fooPi.o -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s
## Local defined symbols.
# RUN: not ld.lld %t/_Z3fooPKi-_Z3fooPi.o -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s

# CONST:      error: undefined symbol: foo(int*)
# CONST-NEXT: >>> referenced by {{.*}}
# CONST-NEXT: >>> did you mean: foo(int const*)

## Case mismatch.
# RUN: not ld.lld %t/test.o %t/_Z3FOOPKi.o -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s
# RUN: not ld.lld %t/_Z3fooPKi-_Z3FOOPKi.o -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s

# CASE:      error: undefined symbol: FOO(int const*)
# CASE-NEXT: >>> referenced by {{.*}}
# CASE-NEXT: >>> did you mean: foo(int const*)

#--- test.s
.globl _start, abcde, _Z3fooPKi
_start:
abcde:
_Z3fooPKi:

#--- bcde-abcd-abde.s
call bcde
call abcd
call abde

#--- bbcde-abcdd.s
call bbcde
call abcdd

#--- aabcde-abcdee.s
call aabcde
call abcdee

#--- bacde.s
call bacde

#--- _Z3fooPi.s
call _Z3fooPi

#--- _Z3fooPKi-_Z3fooPi.s
_Z3fooPKi: call _Z3fooPi

#--- _Z3FOOPKi.s
call _Z3FOOPKi

#--- _Z3fooPKi-_Z3FOOPKi.s
_Z3fooPKi: call _Z3FOOPKi
