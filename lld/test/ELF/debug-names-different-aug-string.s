# REQUIRES: x86
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: sed 's/LLVM0700/LLVM9999/' %S/Inputs/debug-names-a.s | llvm-mc -filetype=obj -triple=x86_64 -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-b.s -o b.o
# RUN: ld.lld --debug-names a.o b.o -o out
# RUN: llvm-dwarfdump -debug-names out | FileCheck %s --check-prefix=DWARF

# DWARF:      .debug_names contents:
# DWARF:      Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length:
# DWARF-NEXT:     Format: DWARF32
# DWARF-NEXT:     Version: 5
# DWARF:          Augmentation: ''
