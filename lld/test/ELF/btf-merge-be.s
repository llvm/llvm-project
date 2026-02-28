# REQUIRES: ppc
## Test --btf-merge with a big-endian target.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux b.s -o b.o
# RUN: ld.lld --btf-merge a.o b.o -o merged
# RUN: llvm-readelf -x .BTF merged | FileCheck %s

## Big-endian BTF magic is 0xeb9f stored as eb9f (not byte-swapped).
# CHECK: Hex dump of section '.BTF':
# CHECK: 0x{{[0-9a-f]+}} eb9f0100

#--- a.s
.text
.globl _start
_start:
  blr

.section .BTF,"",@progbits
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 5                 # str_len
## Type 1: INT "int" size=4
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 4                 # size
.long 0x00000020        # encoding: bits=32
## String table: "\0int\0"
.byte 0
.ascii "int"
.byte 0

#--- b.s
.text
.globl bar
.type bar, @function
bar:
  blr

.section .BTF,"",@progbits
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 6                 # str_len
## Type 1: INT "long" size=8
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 8                 # size
.long 0x00000040        # encoding: bits=64
## String table: "\0long\0"
.byte 0
.ascii "long"
.byte 0
