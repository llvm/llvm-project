# REQUIRES: x86

# FIXME: this should be supported on Windows as well. There is a strange issue
# happening with command line processing though. The command line argument
#   --export-dynamic-symbol 'f*'
# does not have the single quotes stripped on some Windows targets (but not
# all). This causes the glob matching to fail, which means the test fails on
# some Windows bots and passes on others. However, there's no clear indication
# as to what's changed to cause this behavior. Marking the test as unsupported
# so that we have time to investigate the issue without losing postcommit CI.
# UNSUPPORTED: system-windows

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## For an executable, --export-dynamic-symbol exports a symbol if it is non-local and defined.
# RUN: ld.lld -pie --export-dynamic-symbol foo --export-dynamic-symbol qux %t.o -o %t
# RUN: llvm-nm -D -p %t | FileCheck %s
# RUN: echo '{ foo; };' > %t1.list
# RUN: echo '{ foo; qux; };' > %t2.list
# RUN: ld.lld -pie --export-dynamic-symbol-list=%t2.list %t.o -o %t
# RUN: llvm-nm -D -p %t | FileCheck %s

## --export-dynamic exports all non-local defined symbols.
## --export-dynamic-symbol is shadowed.
# RUN: ld.lld -pie --export-dynamic --export-dynamic-symbol foo %t.o -o %t.start
# RUN: llvm-nm -D -p %t.start | FileCheck --check-prefixes=CHECK,START %s

# CHECK-NOT:  .
# START:      T _start
# CHECK:      T foo
# CHECK:      T qux
# CHECK-NOT:  .

## --export-dynamic-symbol does not imply -u: %t1.a(%t1.o) is not fetched.
## This is compatible with GNU ld since binutils 2.35 onwards.
# RUN: echo '.globl foo, bar; foo: bar:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: rm -f %t1.a && llvm-ar rc %t1.a %t1.o
# RUN: ld.lld --export-dynamic-symbol bar %t1.a %t.o -o %t.nofetch
# RUN: llvm-nm %t.nofetch | FileCheck /dev/null --implicit-check-not=bar

## For -shared, if no option expresses a symbolic intention, --export-dynamic-symbol is a no-op.
# RUN: ld.lld -shared --export-dynamic-symbol foo %t.o -o %t.noop
# RUN: llvm-objdump -d %t.noop | FileCheck --check-prefix=PLT2 %s
# RUN: ld.lld -shared --export-dynamic-symbol-list %t2.list %t.o -o %t.noop
# RUN: llvm-objdump -d %t.noop | FileCheck --check-prefix=PLT2 %s

## --export-dynamic-symbol can make a symbol preemptible even if it would be otherwise
## non-preemptible (due to -Bsymbolic, -Bsymbolic-functions or --dynamic-list).
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol nomatch %t.o -o %t.nopreempt
# RUN: llvm-objdump -d %t.nopreempt | FileCheck --check-prefix=NOPLT %s
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol foo %t.o -o %t.preempt
# RUN: llvm-objdump -d %t.preempt | FileCheck --check-prefix=PLT1 %s
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol-list %t1.list %t.o -o %t.preempt
# RUN: llvm-objdump -d %t.preempt | FileCheck --check-prefix=PLT1 %s

## 'nomatch' does not match any symbol. Don't warn.
# RUN: ld.lld --fatal-warnings -shared -Bsymbolic-functions --export-dynamic-symbol nomatch %t.o -o %t.nopreempt2
# RUN: llvm-objdump -d %t.nopreempt2 | FileCheck --check-prefix=NOPLT %s
# RUN: ld.lld -shared -Bsymbolic-functions --export-dynamic-symbol foo %t.o -o %t.preempt2
# RUN: llvm-objdump -d %t.preempt2 | FileCheck --check-prefix=PLT1 %s

# RUN: echo '{};' > %t.list
# RUN: ld.lld -shared --dynamic-list %t.list --export-dynamic-symbol foo %t.o -o %t.preempt3
# RUN: llvm-objdump -d %t.preempt3 | FileCheck --check-prefix=PLT1 %s

## The option value is a glob.
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol 'f*' %t.o -o - | \
# RUN:   llvm-objdump -d - | FileCheck --check-prefix=PLT1 %s
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol '[f]o[o]' %t.o -o - | \
# RUN:   llvm-objdump -d - | FileCheck --check-prefix=PLT1 %s
# RUN: ld.lld -shared -Bsymbolic --export-dynamic-symbol 'f?o' %t.o -o - | \
# RUN:   llvm-objdump -d - | FileCheck --check-prefix=PLT1 %s

# PLT1:      <foo@plt>
# PLT1:      <qux>

# PLT2:      <foo@plt>
# PLT2:      <qux@plt>

# NOPLT-NOT: <foo@plt>
# NOPLT-NOT: <qux@plt>

.global _start, foo, qux
.type foo, @function
.type qux, @function
_start:
  call foo
  call qux
foo:
qux:
