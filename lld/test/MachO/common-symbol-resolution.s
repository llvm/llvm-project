# REQUIRES: x86
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/common.s -o %t/common.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-common.s -o %t/weak-common.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/defined.s -o %t/defined.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-defined.s -o %t/weak-defined.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/calls-foo.s -o %t/calls-foo.o

# RUN: %lld -lSystem -order_file %t/order -dylib %t/libfoo.o -o %t/libfoo.dylib

# RUN: rm -f %t/defined.a %t/weak-defined-and-common.a
# RUN: llvm-ar rcs %t/defined.a %t/defined.o
# RUN: llvm-ar rcs %t/weak-defined-and-common.a %t/weak-defined.o %t/common.o

## The weak attribute appears to have no effect on common symbols. Given two
## common symbols of the same name, we always pick the one with the larger size,
## regardless of whether it is weak. Moreover, the resolved symbol in the output
## file will always be non-weak, even if the winning input symbol definition was
## weak.
# RUN: %lld -lSystem -order_file %t/order %t/common.o %t/weak-common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON
# RUN: %lld -lSystem -order_file %t/order %t/weak-common.o %t/common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON

## Defined symbols are the only ones that take precedence over common symbols.
# RUN: %lld -lSystem -order_file %t/order %t/defined.o %t/common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED
# RUN: %lld -lSystem -order_file %t/order %t/common.o %t/defined.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED

# RUN: %lld -lSystem -order_file %t/order %t/weak-defined.o %t/common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
# RUN: %lld -lSystem -order_file %t/order %t/common.o %t/weak-defined.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED

## Common symbols take precedence over archive symbols.
# RUN: %lld -lSystem -order_file %t/order %t/defined.a %t/weak-common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON
# RUN: %lld -lSystem -order_file %t/order %t/weak-common.o %t/defined.a %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON

## If an archive has both a common and a defined symbol, the defined one should
## win.
# RUN: %lld -lSystem -order_file %t/order %t/weak-defined-and-common.a %t/calls-foo.o -o %t/calls-foo
# RUN: llvm-objdump --syms %t/calls-foo | FileCheck %s --check-prefix=WEAK-DEFINED
# RUN: %lld -lSystem -order_file %t/order %t/calls-foo.o %t/weak-defined-and-common.a -o %t/calls-foo
# RUN: llvm-objdump --syms %t/calls-foo | FileCheck %s --check-prefix=WEAK-DEFINED

## Common symbols take precedence over dylib symbols.
# RUN: %lld -lSystem -order_file %t/order %t/libfoo.dylib %t/weak-common.o %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON
# RUN: %lld -lSystem -order_file %t/order %t/weak-common.o %t/libfoo.dylib %t/test.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=LARGER-COMMON

# LARGER-COMMON-LABEL: SYMBOL TABLE:
# LARGER-COMMON-DAG:   [[#%x, FOO_ADDR:]] g     O __DATA,__common _foo
# LARGER-COMMON-DAG:   [[#FOO_ADDR + 2]]  g     O __DATA,__common _foo_end

# DEFINED-LABEL:       SYMBOL TABLE:
# DEFINED:             g     F __TEXT,__text _foo

# WEAK-DEFINED-LABEL:  SYMBOL TABLE:
# WEAK-DEFINED:        w     F __TEXT,__text _foo

#--- order
## %t/order is important as we determine the size of a given symbol via the
## address of the next symbol.
_foo
_foo_end

#--- common.s
.comm _foo, 1

.globl _bar
_bar:

#--- weak-common.s
.weak_definition _foo
.comm _foo, 2

#--- defined.s
.globl _foo
_foo:
  .quad 0x1234

#--- weak-defined.s
.globl _foo
.weak_definition _foo
_foo:
  .quad 0x1234

#--- libfoo.s
.globl _foo
_foo:
  .quad 0x1234

#--- test.s
.comm _foo_end, 1

.globl _main
_main:
  ret

#--- calls-foo.s
.comm _foo_end, 1

.globl _main
_main:
  callq _foo
  ret
