# REQUIRES: x86

## This tests that we can link against these synthetic symbols even
## if they are not in the symbol table.

# RUN: rm -rf %t; split-file %s %t

## Test that in a dylib, we can link against __mh_dylib_header
## (but not in other types of files)

# RUN: llvm-mc %t/dylib.s -triple=x86_64-apple-macos10.15 -filetype=obj -o %t/dylib.o
# RUN: %lld -dylib -dead_strip %t/dylib.o -o %t/dylib.out
# RUN: llvm-objdump -m --syms %t/dylib.out | FileCheck %s --check-prefix DYLIB

# RUN: not %lld -o /dev/null %t/dylib.o 2>&1 | FileCheck %s --check-prefix ERR-DYLIB

# DYLIB:      SYMBOL TABLE:
# DYLIB-NEXT: {{[0-9a-f]+}} g     F __TEXT,__text _main
# DYLIB-NEXT-EMPTY:
# ERR-DYLIB: error: undefined symbol: __mh_dylib_header

## Test that in an executable, we can link against __mh_execute_header
# RUN: llvm-mc %t/main.s -triple=x86_64-apple-macos10.15 -filetype=obj -o %t/exec.o
# RUN: %lld -dead_strip -lSystem %t/exec.o -o %t/exec.out

## But it would be an error trying to reference __mh_execute_header in a dylib
# RUN: not %lld -o /dev/null -dylib %t/exec.o 2>&1 | FileCheck %s --check-prefix ERR-EXEC

# ERR-EXEC: error: undefined symbol: __mh_execute_header

#--- main.s
.text
.globl _main
_main:
 mov __mh_execute_header@GOTPCREL(%rip), %rax
 ret
.subsections_via_symbols

#--- dylib.s
.text
.globl _main
_main:
 mov __mh_dylib_header@GOTPCREL(%rip), %rax
 ret
.subsections_via_symbols
