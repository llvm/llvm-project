# REQUIRES: x86
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-a.s -o a.o
# RUN: echo '.globl foo; foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o b0.o

# RUN: ld.lld --debug-names a.o b0.o -o out0
# RUN: llvm-dwarfdump -debug-names out0 | FileCheck %s --check-prefix=DWARF

# DWARF:      .debug_names contents:
# DWARF-NEXT: Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length: 0x6F
# DWARF-NEXT:     Format: DWARF32
# DWARF-NEXT:     Version: 5
# DWARF-NEXT:     CU count: 1
# DWARF-NEXT:     Local TU count: 0
# DWARF-NEXT:     Foreign TU count: 0
# DWARF-NEXT:     Bucket count: 2
# DWARF-NEXT:     Name count: 2
# DWARF-NEXT:     Abbreviations table size: 0x15
# DWARF-NEXT:     Augmentation: 'LLVM0700'
# DWARF-NEXT:   }
# DWARF-NEXT:   Compilation Unit offsets [
# DWARF-NEXT:     CU[0]: 0x00000000
# DWARF-NEXT:   ]

## Test both files without .debug_names.
# RUN: echo '.globl _start; _start:' | llvm-mc -filetype=obj -triple=x86_64 - -o a0.o
# RUN: ld.lld --debug-names a0.o b0.o -o out1
# RUN: llvm-readelf -SW out1 | FileCheck %s --check-prefix=ELF

# ELF:      Name              Type     Address          Off      Size   ES Flg Lk Inf Al
# ELF-NOT: .debug_names
