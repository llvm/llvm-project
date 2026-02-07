# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 test.s -o test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bcde-abcd-abde.s -o bcde-abcd-abde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bbcde-abcdd.s -o bbcde-abcdd.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 aabcde-abcdee.s -o aabcde-abcdee.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bacde.s -o bacde.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 _Z3fooPi.s -o _Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 _Z3fooPKi-_Z3fooPi.s -o _Z3fooPKi-_Z3fooPi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 _Z3FOOPKi.s -o _Z3FOOPKi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 _Z3fooPKi-_Z3FOOPKi.s -o _Z3fooPKi-_Z3FOOPKi.o

## Insert a character.
## The spell corrector is enabled for the first two "undefined symbol" diagnostics.
# RUN: not ld.lld test.o bcde-abcd-abde.o 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=test.o --implicit-check-not=error:

## Symbols defined in DSO can be suggested.
# RUN: ld.lld test.o -shared -o test.so
# RUN: not ld.lld test.so bcde-abcd-abde.o 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=test.so --implicit-check-not=error:

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

## Substitute a character.
# RUN: not ld.lld test.o bbcde-abcdd.o 2>&1 | FileCheck --check-prefix=SUBST %s --implicit-check-not=error:

# SUBST:      error: undefined symbol: bbcde
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde
# SUBST:      error: undefined symbol: abcdd
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde

## Delete a character.
# RUN: not ld.lld test.o aabcde-abcdee.o 2>&1 | FileCheck --check-prefix=DELETE %s --implicit-check-not=error:

# DELETE:      error: undefined symbol: aabcde
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde
# DELETE:      error: undefined symbol: abcdee
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde

## Transpose.
# RUN: not ld.lld test.o bacde.o 2>&1 | FileCheck --check-prefix=TRANSPOSE %s --implicit-check-not=error:

# TRANSPOSE:      error: undefined symbol: bacde
# TRANSPOSE-NEXT: >>> referenced by {{.*}}
# TRANSPOSE-NEXT: >>> did you mean: abcde

## Missing const qualifier.
# RUN: not ld.lld test.o _Z3fooPi.o 2>&1 | FileCheck --check-prefix=CONST %s --implicit-check-not=error:
## Local defined symbols.
# RUN: not ld.lld _Z3fooPKi-_Z3fooPi.o 2>&1 | FileCheck --check-prefix=CONST %s --implicit-check-not=error:

# CONST:      error: undefined symbol: foo(int*)
# CONST-NEXT: >>> referenced by {{.*}}
# CONST-NEXT: >>> did you mean: foo(int const*)

## Case mismatch.
# RUN: not ld.lld test.o _Z3FOOPKi.o 2>&1 | FileCheck --check-prefix=CASE %s --implicit-check-not=error:
# RUN: not ld.lld _Z3fooPKi-_Z3FOOPKi.o 2>&1 | FileCheck --check-prefix=CASE %s --implicit-check-not=error:

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
