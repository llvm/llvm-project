# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o

## Insert a character.
## The spell corrector is enabled for the first two "undefined symbol" diagnostics.
# RUN: echo 'call bcde; call abcd; call abde' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t.o

## Symbols defined in DSO can be suggested.
# RUN: %lld %t.o -dylib -o %t.dylib
# RUN: not %lld %t.dylib %t1.o -o /dev/null 2>&1 | FileCheck --check-prefix=INSERT %s -DFILE=%t.dylib

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
# RUN: echo 'call bbcde; call abcdd' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck --check-prefix=SUBST %s

# SUBST:      error: undefined symbol: abcdd
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde
# SUBST:      error: undefined symbol: bbcde
# SUBST-NEXT: >>> referenced by {{.*}}
# SUBST-NEXT: >>> did you mean: abcde

## Delete a character.
# RUN: echo 'call aabcde; call abcdee' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck --check-prefix=DELETE %s

# DELETE:      error: undefined symbol: abcdee
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde
# DELETE:      error: undefined symbol: aabcde
# DELETE-NEXT: >>> referenced by {{.*}}
# DELETE-NEXT: >>> did you mean: abcde

## Transpose.
# RUN: echo 'call bacde' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck --check-prefix=TRANSPOSE %s

# TRANSPOSE:      error: undefined symbol: bacde
# TRANSPOSE-NEXT: >>> referenced by {{.*}}
# TRANSPOSE-NEXT: >>> did you mean: abcde

## Missing const qualifier.
# RUN: echo 'call __Z3fooPi' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s
## Local defined symbols.
# RUN: echo '__Z3fooPKi: call __Z3fooPi' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t1.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CONST %s

# CONST:      error: undefined symbol: foo(int*)
# CONST-NEXT: >>> referenced by {{.*}}
# CONST-NEXT: >>> did you mean: foo(int const*)

## Case mismatch.
# RUN: echo 'call __Z3FOOPKi' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s
# RUN: echo '__Z3fooPKi: call __Z3FOOPKi' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t1.o -demangle -o /dev/null 2>&1 | FileCheck --check-prefix=CASE %s

# CASE:      error: undefined symbol: FOO(int const*)
# CASE-NEXT: >>> referenced by {{.*}}
# CASE-NEXT: >>> did you mean: foo(int const*)

.globl _main, abcde, __Z3fooPKi
_main:
abcde:
__Z3fooPKi:
