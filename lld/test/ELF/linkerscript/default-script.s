# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld --default-script=def.t b.t -T a.t a.o -o out
# RUN: llvm-readelf -Ss out | FileCheck %s

# CHECK:      Name
# CHECK:      .foo2
# CHECK-NEXT: .foo0
# CHECK-NEXT: .foo1
# CHECK:      1: 000000000000000c     0 NOTYPE  GLOBAL DEFAULT     4 _start
# CHECK-NEXT: 2: 000000000000002a     0 NOTYPE  GLOBAL DEFAULT   ABS b
# CHECK-NEXT: 3: 000000000000002a     0 NOTYPE  GLOBAL DEFAULT   ABS a
# CHECK-EMPTY:

## In the absence of --script options, the default linker script is read.
# RUN: ld.lld --default-script def.t b.t a.o -o out1
# RUN: llvm-readelf -Ss out1 | FileCheck %s --check-prefix=CHECK1
# RUN: ld.lld -dT def.t b.t a.o -o out1a && cmp out1 out1a
## If multiple -dT options are specified, the last -dT wins.
# RUN: ld.lld -dT a.t -dT def.t b.t a.o -o out1a && cmp out1 out1a

# RUN: mkdir d && cp def.t d/default.t
# RUN: ld.lld -L d -dT default.t b.t a.o -o out1a && cmp out1 out1a

# CHECK1:      Name
# CHECK1:      .foo2
# CHECK1-NEXT: .foo1
# CHECK1-NEXT: .foo0
# CHECK1:      1: 000000000000000c     0 NOTYPE  GLOBAL DEFAULT     4 _start
# CHECK1-NEXT: 2: 000000000000002a     0 NOTYPE  GLOBAL DEFAULT   ABS b
# CHECK1-NEXT: 3: 000000000000002a     0 NOTYPE  GLOBAL DEFAULT   ABS def
# CHECK1-EMPTY:

# RUN: not ld.lld --default-script not-exist.t b.t -T a.t a.o 2>&1 | FileCheck %s --check-prefix=ERR
# ERR: error: cannot find linker script not-exist.t

#--- a.s
.globl _start
_start:

.section .foo0,"a"; .long 0
.section .foo1,"a"; .long 0
.section .foo2,"a"; .long 0

#--- a.t
a = 42;
SECTIONS {
  .foo2 : {}
  .foo0 : {}
  .foo1 : {}
}

#--- b.t
b = 42;

#--- def.t
def = 42;
SECTIONS {
  .foo2 : {}
  .foo1 : {}
  .foo0 : {}
}
