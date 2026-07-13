# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -machine:x64 -def:%S/Inputs/library.def -implib:%t.lib
# RUN: lld-link -out:%t.exe -entry:main %t.obj %t.lib -debug:dwarf
# RUN: llvm-readobj --string-table %t.exe | FileCheck %s
# RUN: llvm-nm %t.exe | FileCheck %s --check-prefix=SYMBOLS

# Note, for this test to have the intended test coverage, the imported symbol
# "function" needs to be such that the symbol name itself is <= 8 chars, while
# "__imp_"+name is >8 chars.

# CHECK:      StringTable {
# CHECK-NEXT:   Length: 102
# CHECK-NEXT:   [     4] .debug_abbrev
# CHECK-NEXT:   [    12] .debug_line
# CHECK-NEXT:   [    1e] long_name_symbolz
# CHECK-NEXT:   [    30] .debug_abbrez
# CHECK-NEXT:   [    3e] __imp_function
# CHECK-NEXT:   [    4d] __impl_long_name_symbolA
# CHECK-NEXT: }

# SYMBOLS:      140001000 N .debug_abbrez
# SYMBOLS-NEXT: 140002070 R __imp_function
# SYMBOLS-NEXT: 140001000 t __impl_long_name_symbolA
# SYMBOLS-NEXT: 140001010 T function
# SYMBOLS-NEXT: 140001000 t long_name_symbolA
# SYMBOLS-NEXT: 140001000 t long_name_symbolz
# SYMBOLS-NEXT: 140001000 T main
# SYMBOLS-NEXT: 140001000 t name_symbolA

.global main
.text
main:
long_name_symbolz:
long_name_symbolA:
__impl_long_name_symbolA:
name_symbolA:
.debug_abbrez:
  call function
  ret

.section        .debug_abbrev,"dr"
.byte 0

.section        .debug_line,"dr"
.byte 0
