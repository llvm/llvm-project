# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This test demonstrates that when we have two weak symbols of the same type,
## we pick the one whose containing file appears earlier in the command-line
## invocation.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak1.s -o %t/weak1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak2.s -o %t/weak2.o

# RUN: %lld -lSystem -o %t/obj12 -L%t %t/weak1.o %t/weak2.o %t/test.o
# RUN: llvm-objdump --syms %t/obj12 | FileCheck %s --check-prefix=WEAK1
# RUN: %lld -lSystem -o %t/obj21 -L%t %t/weak2.o %t/weak1.o %t/test.o
# RUN: llvm-objdump --syms %t/obj21 | FileCheck %s --check-prefix=WEAK2

# WEAK1: O __TEXT,weak1 _foo
# WEAK2: O __TEXT,weak2 _foo

# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libweak1.dylib %t/weak1.o -o %t/libweak1.dylib
# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libweak2.dylib %t/weak2.o -o %t/libweak2.dylib

# RUN: %lld -lSystem -o %t/dylib12 -L%t -lweak1 -lweak2 %t/test.o
# RUN: llvm-objdump --macho --bind %t/dylib12 | FileCheck %s --check-prefix=DYLIB1
# RUN: %lld -lSystem -o %t/dylib21 -L%t -lweak2 -lweak1 %t/test.o
# RUN: llvm-objdump --macho --bind %t/dylib21 | FileCheck %s --check-prefix=DYLIB2
# DYLIB1: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} pointer 0 libweak1         _foo
# DYLIB2: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} pointer 0 libweak2         _foo

#--- weak1.s
.globl _foo
.weak_definition _foo
.section __TEXT,weak1;
_foo:

#--- weak2.s
.globl _foo
.weak_definition _foo
.section __TEXT,weak2
_foo:

#--- test.s
.globl _main
_main:
  callq _foo
  ret
