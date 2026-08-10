# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/c.s -o %t/c.o
# RUN: ld.lld -shared -soname=b -Ttext=0 %t/b.o -o %t/b.so

# RUN: ld.lld %t/a.o %t/b.so -o %t1
# RUN: llvm-readelf -r -s %t1 | FileCheck %s

## In %t/b.so, foo has st_value==0 and its section alignment is 0x400. The
## alignment of the copy relocated foo is thus 0x400.
# CHECK: R_X86_64_COPY {{.*}} foo + 0
# CHECK: 0000000000203400  4 OBJECT GLOBAL DEFAULT [[#]] foo

## Error if attempting to copy relocate a SHN_ABS symbol (even if st_size is non-zero).
# RUN: ld.lld -shared -soname=c %t/c.o -o %t/c.so
# RUN: llvm-readelf -s %t/c.so | FileCheck %s --check-prefix=ABSADDR
# RUN: not ld.lld %t/a.o %t/c.so -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ABSADDR: 0000000000000000  4 OBJECT GLOBAL DEFAULT ABS foo
# ERR: error: cannot create a copy relocation for symbol foo

#--- a.s
.text
.globl _start
_start:
  movl $5, foo

#--- b.s
.data
.balign 0x400
.type foo,@object
.globl foo
foo:
  .long 0
  .size foo, 4

#--- c.s
.data
.globl foo
.type foo,@object
foo = 0x0
.size foo, 4
