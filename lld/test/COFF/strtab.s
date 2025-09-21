# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -out:%t.exe -entry:main %t.obj -debug:dwarf
# RUN: llvm-readobj --string-table %t.exe | FileCheck %s

# CHECK:      StringTable {
# CHECK-NEXT:   Length: 87
# CHECK-NEXT:   [     4] .debug_abbrev
# CHECK-NEXT:   [    12] .debug_line
# CHECK-NEXT:   [    1e] long_name_symbolz
# CHECK-NEXT:   [    30] .debug_abbrez
# CHECK-NEXT:   [    3e] __impl_long_name_symbolA
# CHECK-NEXT: }


.global main
.text
main:
long_name_symbolz:
long_name_symbolA:
__impl_long_name_symbolA:
name_symbolA:
.debug_abbrez:
  ret

.section        .debug_abbrev,"dr"
.byte 0

.section        .debug_line,"dr"
.byte 0
