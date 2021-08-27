# REQUIRES: x86
# RUN: rm -fr %t && split-file %s %t

## Build an object with a trivial main function
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t1.o

## Build %t.a which defines a global 'foo'
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/archive.s -o %t2.o
# RUN: rm -f %t2.a
# RUN: llvm-ar rc %t2.a %t2.o

## Build %t.so that has a reference to 'foo'
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/shlib.s -o %t3.o
# RUN: ld.lld %t3.o -o %t3.so -shared

## Test that 'foo' from %t2.a is fetched to define 'foo' needed by %t3.so.
## Test both cases where the archive is before or after the shared library in
## link order.

# RUN: ld.lld %t1.o %t2.a %t3.so -o %t.exe
# RUN: llvm-readelf --dyn-symbols %t.exe | FileCheck %s --check-prefix=CHECK-FETCH

# RUN: ld.lld %t1.o %t3.so %t2.a -o %t.exe
# RUN: llvm-readelf --dyn-symbols %t.exe | FileCheck %s --check-prefix=CHECK-FETCH

# CHECK-FETCH: GLOBAL DEFAULT {{[0-9]+}} foo

#--- main.s
.text
.globl _start
.type _start,@function
_start:
  ret

#--- archive.s
.global foo
foo:

#--- shlib.s
.global foo
