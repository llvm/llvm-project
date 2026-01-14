# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin  %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/more-cstrings.s -o %t/more-cstrings.o

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/test-0 %t/test.o %t/more-cstrings.o
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-0 | FileCheck %s --check-prefix=ORIGIN_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-0 | FileCheck %s --check-prefix=ORIGIN_SEC

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/test-1 %t/test.o %t/more-cstrings.o -order_file %t/ord-1
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-1 | FileCheck %s --check-prefix=ONE_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-1 | FileCheck %s --check-prefix=ONE_SEC

# RUN: %lld --no-deduplicate-strings -arch arm64 -lSystem -e _main -o %t/test-1-dup %t/test.o %t/more-cstrings.o -order_file %t/ord-1
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-1-dup | FileCheck %s --check-prefix=ONE_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-1-dup | FileCheck %s --check-prefix=ONE_SEC

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/test-2 %t/test.o %t/more-cstrings.o -order_file %t/ord-2
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-2 | FileCheck %s --check-prefix=TWO_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-2 | FileCheck %s --check-prefix=TWO_SEC

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/test-3 %t/test.o %t/more-cstrings.o -order_file %t/ord-3
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-3 | FileCheck %s --check-prefix=THREE_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-3 | FileCheck %s --check-prefix=THREE_SEC

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/test-4 %t/test.o %t/more-cstrings.o -order_file %t/ord-4
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/test-4 | FileCheck %s --check-prefix=FOUR_SYM
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/test-4 | FileCheck %s --check-prefix=FOUR_SEC
# RUN: llvm-readobj --string-dump=__cstring %t/test-4 | FileCheck %s --check-prefix=FOUR_SEC_ESCAPE

# We expect:
# 1) Covered cstring symbols to be reordered
# 2) the rest of the cstring symbols remain in the original relative order within the cstring section

# ORIGIN_SYM: _local_foo1
# ORIGIN_SYM: _globl_foo2
# ORIGIN_SYM: _local_foo2
# ORIGIN_SYM: _bar
# ORIGIN_SYM: _baz
# ORIGIN_SYM: _baz_dup
# ORIGIN_SYM: _bar2
# ORIGIN_SYM: _globl_foo3

# ORIGIN_SEC: foo1
# ORIGIN_SEC: foo2
# ORIGIN_SEC: bar
# ORIGIN_SEC: baz
# ORIGIN_SEC: bar2
# ORIGIN_SEC: foo3

# original order, but only parital covered
#--- ord-1
#foo2
CSTR;1433942677
#bar
CSTR;540201826
#bar2
CSTR;1496286555
#foo3
CSTR;1343999025

# ONE_SYM-DAG: _globl_foo2
# ONE_SYM-DAG: _local_foo2
# ONE_SYM: _bar
# ONE_SYM: _bar2
# ONE_SYM: _globl_foo3
# ONE_SYM: _local_foo1
# ONE_SYM: _baz
# ONE_SYM: _baz_dup

# ONE_SEC: foo2
# ONE_SEC: bar
# ONE_SEC: bar2
# ONE_SEC: foo3
# ONE_SEC: foo1
# ONE_SEC: baz


# TWO_SYM: _globl_foo2
# TWO_SYM: _local_foo2
# TWO_SYM: _local_foo1
# TWO_SYM: _baz
# TWO_SYM: _baz_dup
# TWO_SYM: _bar
# TWO_SYM: _bar2
# TWO_SYM: _globl_foo3

# TWO_SEC: foo2
# TWO_SEC: foo1
# TWO_SEC: baz
# TWO_SEC: bar
# TWO_SEC: bar2
# TWO_SEC: foo3


# THREE_SYM: _local_foo1
# THREE_SYM: _baz
# THREE_SYM: _baz_dup
# THREE_SYM: _bar
# THREE_SYM: _bar2
# THREE_SYM: _globl_foo2
# THREE_SYM: _local_foo2
# THREE_SYM: _globl_foo3

# THREE_SEC: foo1
# THREE_SEC: baz
# THREE_SEC: bar
# THREE_SEC: bar2
# THREE_SEC: foo2
# THREE_SEC: foo3


# FOUR_SYM: _local_escape_white_space
# FOUR_SYM: _globl_foo2
# FOUR_SYM: _local_foo2
# FOUR_SYM: _local_escape
# FOUR_SYM: _globl_foo3
# FOUR_SYM: _bar
# FOUR_SYM: _local_foo1
# FOUR_SYM: _baz
# FOUR_SYM: _baz_dup
# FOUR_SYM: _bar2

# FOUR_SEC: \t\n
# FOUR_SEC: foo2
# FOUR_SEC: @\"NSDictionary\"
# FOUR_SEC: foo3
# FOUR_SEC: bar
# FOUR_SEC: foo1
# FOUR_SEC: baz
# FOUR_SEC: bar2

# FOUR_SEC_ESCAPE: ..
# FOUR_SEC_ESCAPE: foo2
# FOUR_SEC_ESCAPE: @"NSDictionary"
# FOUR_SEC_ESCAPE: foo3
# FOUR_SEC_ESCAPE: bar
# FOUR_SEC_ESCAPE: foo1
# FOUR_SEC_ESCAPE: baz
# FOUR_SEC_ESCAPE: bar2


# change order, parital covered
#--- ord-2
#foo2
CSTR;1433942677
#foo1
CSTR;1663475769
#baz
CSTR;862947621
#bar
CSTR;540201826
#bar2
CSTR;1496286555

# change order, parital covered, with mismatches, duplicates
#--- ord-3
foo2222
CSTR;0x11111111
#bar (mismatched cpu and file name)
fakeCPU:fake-file-name.o:CSTR;540201826
#not a hash
CSTR;xxx
#foo1
CSTR;1663475769
#baz
CSTR;862947621
#bar
CSTR;540201826
#bar2
CSTR;1496286555
#baz
CSTR;862947621

# test escape strings
#--- ord-4
#\t\n
CSTR;1035903177
#foo2
CSTR;1433942677
#@\"NSDictionary\"
CSTR;1202669430
#foo3
CSTR;1343999025
#bar
CSTR;540201826


#--- test.s
.text
.globl _main

_main:
  ret

.cstring
.p2align 2
_local_foo1:
  .asciz "foo1"
_local_foo2:
  .asciz "foo2"
L_.foo1_dup:
  .asciz "foo1"
L_.foo2_dup:
  .asciz "foo2"
_local_escape:
  .asciz "@\"NSDictionary\""
_local_escape_white_space:
  .asciz "\t\n"

_bar:
  .asciz "bar"
_baz:
  .asciz "baz"
_bar2:
  .asciz "bar2"
_baz_dup:
  .asciz "baz"

.subsections_via_symbols

#--- more-cstrings.s
.globl _globl_foo1, _globl_foo3
.cstring
.p2align 4
_globl_foo3:
  .asciz "foo3"
_globl_foo2:
  .asciz "foo2"
