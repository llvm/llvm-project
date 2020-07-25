# REQUIRES: x86
# RUN: mkdir -p %t

## This test demonstrates that when we have two weak symbols of the same type,
## we pick the one whose containing file appears earlier in the command-line
## invocation.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: echo ".globl _foo; .weak_definition _foo; .section __TEXT,weak1; _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/weak1.o
# RUN: echo ".globl _foo; .weak_definition _foo; .section __TEXT,weak2; _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/weak2.o

# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/obj12 -Z -L%t %t/weak1.o %t/weak2.o %t/test.o
# RUN: llvm-objdump --syms %t/obj12 | FileCheck %s --check-prefix=WEAK1
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/obj21 -Z -L%t %t/weak2.o %t/weak1.o %t/test.o
# RUN: llvm-objdump --syms %t/obj21 | FileCheck %s --check-prefix=WEAK2

# WEAK1: O __TEXT,weak1 _foo
# WEAK2: O __TEXT,weak2 _foo

# RUN: lld -flavor darwinnew -dylib -install_name \
# RUN:   @executable_path/libweak1.dylib %t/weak1.o -o %t/libweak1.dylib
# RUN: lld -flavor darwinnew -dylib -install_name \
# RUN:   @executable_path/libweak2.dylib %t/weak2.o -o %t/libweak2.dylib

# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/dylib12 -Z -L%t -lweak1 -lweak2 %t/test.o
# RUN: llvm-objdump --macho --lazy-bind %t/dylib12 | FileCheck %s --check-prefix=DYLIB1
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t/dylib21 -Z -L%t -lweak2 -lweak1 %t/test.o
# RUN: llvm-objdump --macho --lazy-bind %t/dylib21 | FileCheck %s --check-prefix=DYLIB2
## TODO: these should really be in the weak binding section, not the lazy binding section
# DYLIB1: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} libweak1         _foo
# DYLIB2: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} libweak2         _foo

.globl _main
_main:
  callq _foo
  ret
