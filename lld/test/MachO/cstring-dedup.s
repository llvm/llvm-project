# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/more-foo.s -o %t/more-foo.o
# RUN: %lld -dylib %t/test.o %t/more-foo.o -o %t/test
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test | \
# RUN:   FileCheck %s --check-prefix=STR --implicit-check-not foo --implicit-check-not bar
# RUN: llvm-objdump --macho --section="__DATA,ptrs" --syms %t/test | FileCheck %s
# RUN: llvm-readobj --section-headers %t/test | FileCheck %s --check-prefix=HEADER

## Make sure we only have 3 deduplicated strings in __cstring.
# STR: Contents of (__TEXT,__cstring) section
# STR: {{[[:xdigit:]]+}} foo
# STR: {{[[:xdigit:]]+}} barbaz
# STR: {{[[:xdigit:]]+}} {{$}}

## Make sure both symbol and section relocations point to the right thing.
# CHECK:      Contents of (__DATA,ptrs) section
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:foo
# CHECK-NEXT: __TEXT:__cstring:barbaz
# CHECK-NEXT: __TEXT:__cstring:baz
# CHECK-NEXT: __TEXT:__cstring:barbaz
# CHECK-NEXT: __TEXT:__cstring:baz
# CHECK-NEXT: __TEXT:__cstring:{{$}}
# CHECK-NEXT: __TEXT:__cstring:{{$}}

## Make sure the symbol addresses are correct too.
# CHECK:     SYMBOL TABLE:
# CHECK-DAG: [[#%.16x,FOO:]]  l     O __TEXT,__cstring _local_foo1
# CHECK-DAG: [[#FOO]]         l     O __TEXT,__cstring _local_foo2
# CHECK-DAG: [[#FOO]]         g     O __TEXT,__cstring _globl_foo1
# CHECK-DAG: [[#FOO]]         g     O __TEXT,__cstring _globl_foo2
# CHECK-DAG: [[#%.16x,BAR:]]  l     O __TEXT,__cstring _bar1
# CHECK-DAG: [[#BAR]]         l     O __TEXT,__cstring _bar2
# CHECK-DAG: [[#%.16x,ZERO:]] l     O __TEXT,__cstring _zero1
# CHECK-DAG: [[#ZERO]]        l     O __TEXT,__cstring _zero2

## Make sure we set the right alignment and flags.
# HEADER:        Name: __cstring
# HEADER-NEXT:   Segment: __TEXT
# HEADER-NEXT:   Address:
# HEADER-NEXT:   Size:
# HEADER-NEXT:   Offset:
# HEADER-NEXT:   Alignment: 4
# HEADER-NEXT:   RelocationOffset:
# HEADER-NEXT:   RelocationCount: 0
# HEADER-NEXT:   Type: CStringLiterals
# HEADER-NEXT:   Attributes [ (0x0)
# HEADER-NEXT:   ]
# HEADER-NEXT:   Reserved1: 0x0
# HEADER-NEXT:   Reserved2: 0x0
# HEADER-NEXT:   Reserved3: 0x0

#--- test.s
.cstring
.p2align 2
_local_foo1:
  .asciz "foo"
_local_foo2:
  .asciz "foo"
L_.foo1:
  .asciz "foo"
L_.foo2:
  .asciz "foo"

_bar1:
  .ascii "bar"
_baz1:
  .asciz "baz"
_bar2:
  .ascii "bar"
_baz2:
  .asciz "baz"

_zero1:
  .asciz ""
_zero2:
  .asciz ""

.section __DATA,ptrs,literal_pointers
.quad L_.foo1
.quad L_.foo2
.quad _local_foo1
.quad _local_foo2
.quad _globl_foo1
.quad _globl_foo2
.quad _bar1
.quad _baz1
.quad _bar2
.quad _baz2
.quad _zero1
.quad _zero2

#--- more-foo.s
.globl _globl_foo1, _globl_foo2
.cstring
.p2align 4
_globl_foo1:
  .asciz "foo"
_globl_foo2:
  .asciz "foo"
