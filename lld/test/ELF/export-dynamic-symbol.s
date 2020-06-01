# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## For an executable, --export-dynamic-symbol exports a symbol if it is non-local and defined.
# RUN: ld.lld -pie --export-dynamic-symbol foo %t.o -o %t
# RUN: llvm-nm -D -p %t | FileCheck %s

## --export-dynamic exports all non-local defined symbols.
## --export-dynamic-symbol is shadowed.
# RUN: ld.lld -pie --export-dynamic --export-dynamic-symbol foo %t.o -o %t.start
# RUN: llvm-nm -D -p %t.start | FileCheck --check-prefixes=CHECK,START %s

# CHECK-NOT:  .
# START:      T _start
# CHECK:      T foo
# CHECK-NOT:  .

## --export-dynamic-symbol does not imply -u: %t1.a(%t1.o) is not fetched.
## This is compatible with GNU ld since binutils 2.35 onwards.
# RUN: echo '.globl foo, bar; foo: bar:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: rm -f %t1.a && llvm-ar rc %t1.a %t1.o
# RUN: ld.lld --export-dynamic-symbol bar %t1.a %t.o -o %t.nofetch
# RUN: llvm-nm %t.nofetch | FileCheck /dev/null --implicit-check-not=bar

## --export-dynamic-symbol can make a symbol preemptible even if it would be otherwise
## non-preemptible (due to -Bsymbolic, -Bsymbolic-functions or --dynamic-list).
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol nomatch %t.o -o %t.nopreempt
# RUN: llvm-objdump -d %t.nopreempt | FileCheck --check-prefix=NOPLT %s
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol foo %t.o -o %t.preempt
# RUN: llvm-objdump -d %t.preempt | FileCheck --check-prefix=PLT %s

## 'nomatch' does not match any symbol. Don't warn.
# RUN: ld.lld --fatal-warnings -shared -Bsymbolic-functions --export-dynamic-symbol nomatch %t.o -o %t.nopreempt2
# RUN: llvm-objdump -d %t.nopreempt2 | FileCheck --check-prefix=NOPLT %s
# RUN: ld.lld -shared -Bsymbolic-functions --export-dynamic-symbol foo %t.o -o %t.preempt2
# RUN: llvm-objdump -d %t.preempt2 | FileCheck --check-prefix=PLT %s

# RUN: echo '{};' > %t.list
# RUN: ld.lld -shared --dynamic-list %t.list --export-dynamic-symbol foo %t.o -o %t.preempt3
# RUN: llvm-objdump -d %t.preempt3 | FileCheck --check-prefix=PLT %s

## The option value is a glob.
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol 'f*' %t.o -o %t.preempt4
# RUN: llvm-objdump -d %t.preempt4 | FileCheck --check-prefix=PLT %s

# PLT:       <foo@plt>
# NOPLT-NOT: <foo@plt>

.global _start, foo
.type foo, @function
_start:
  call foo
foo:
