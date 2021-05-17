# REQUIRES: x86
## Test that lazy symbols in a section group can be demoted to Undefined,
## so that we can report a "discarded section" error.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl f1, foo; f1: call foo; \
# RUN:   .section .text.foo,"axG",@progbits,foo,comdat; foo:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o

## Test the case when the symbol causing a "discarded section" is ordered
## *before* the symbol fetching the lazy object.
## The test relies on the symbol table order of llvm-mc (lexical), which will
## need adjustment if llvm-mc changes its behavior.
# RUN: echo '.globl aa, f2; f2: call aa; \
# RUN:   .section .text.foo,"axG",@progbits,foo,comdat; aa:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %taa.o
# RUN: llvm-nm -p %taa.o | FileCheck --check-prefix=AA-NM %s
# RUN: not ld.lld %t.o --start-lib %t1.o %taa.o --end-lib -o /dev/null 2>&1 | FileCheck --check-prefix=AA %s
# RUN: rm -f %taa.a && llvm-ar rc %taa.a %taa.o
# RUN: not ld.lld %t.o --start-lib %t1.o --end-lib %taa.a -o /dev/null 2>&1 | FileCheck --check-prefix=AA %s

# AA-NM: aa
# AA-NM: f2

# AA:      error: relocation refers to a symbol in a discarded section: aa
# AA-NEXT: >>> defined in {{.*}}aa.o
# AA-NEXT: >>> section group signature: foo
# AA-NEXT: >>> prevailing definition is in {{.*}}1.o
# AA-NEXT: >>> referenced by {{.*}}aa.o:(.text+0x1)

## Test the case when the symbol causing a "discarded section" is ordered
## *after* the symbol fetching the lazy object.
# RUN: echo '.globl f2, zz; f2: call zz; \
# RUN:   .section .text.foo,"axG",@progbits,foo,comdat; zz:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %tzz.o
# RUN: llvm-nm -p %tzz.o | FileCheck --check-prefix=ZZ-NM %s
# RUN: not ld.lld %t.o --start-lib %t1.o %tzz.o --end-lib -o /dev/null 2>&1 | FileCheck --check-prefix=ZZ %s
# RUN: rm -f %tzz.a && llvm-ar rc %tzz.a %tzz.o
# RUN: not ld.lld %t.o --start-lib %t1.o --end-lib %tzz.a -o /dev/null 2>&1 | FileCheck --check-prefix=ZZ %s

# ZZ-NM: f2
# ZZ-NM: zz

# ZZ:      error: relocation refers to a symbol in a discarded section: zz
# ZZ-NEXT: >>> defined in {{.*}}zz.o
# ZZ-NEXT: >>> section group signature: foo
# ZZ-NEXT: >>> prevailing definition is in {{.*}}1.o
# ZZ-NEXT: >>> referenced by {{.*}}zz.o:(.text+0x1)

## Don't error if the symbol which would cause "discarded section"
## was inserted before %tzz.o
# RUN: echo '.globl zz; zz:' | llvm-mc -filetype=obj -triple=x86_64 - -o %tdef.o
# RUN: ld.lld %t.o --start-lib %t1.o %tdef.o %tzz.o --end-lib -o /dev/null
# RUN: rm -f %tdef.a && llvm-ar rc %tdef.a %tdef.o
# RUN: ld.lld %t.o --start-lib %t1.o %tdef.a %tzz.o --end-lib -o /dev/null

.globl _start
_start:
  call f1
  call f2
