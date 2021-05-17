# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o

## foo and __foo are interconnected and defined in two lazy object files.
## Test we resolve both to the same file.
# RUN: ld.lld -y a -y foo -y __foo %t/main.o --start-lib %t/a.o %t/b.o --end-lib -o /dev/null | FileCheck %s

# CHECK:      a.o: lazy definition of a
# CHECK-NEXT: a.o: lazy definition of foo
# CHECK-NEXT: a.o: lazy definition of __foo
# CHECK-NEXT: b.o: definition of foo
# CHECK-NEXT: b.o: definition of __foo
# CHECK-NEXT: b.o: reference to a
# CHECK-NEXT: a.o: definition of a

#--- main.s
.globl _start
_start:
  call b

#--- a.s
.globl a
.weak foo
a:
foo:

.weak __foo
__foo:

#--- b.s
.globl b
.weak foo
b:
  call a
foo:

.weak __foo
__foo:
