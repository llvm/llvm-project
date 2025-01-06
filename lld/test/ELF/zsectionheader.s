# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared -z nosectionheader -z sectionheader %t.o -o %t.so 2>&1 | count 0
# RUN: llvm-readelf -hS %t.so | FileCheck %s --check-prefixes=CHECK,SHDR

# RUN: ld.lld -shared -z nosectionheader %t.o -o %t0.so
# RUN: llvm-readelf -h --dyn-syms %t0.so | FileCheck %s --check-prefixes=CHECK,NOSHDR
# RUN: llvm-strings %t0.so | FileCheck %s --check-prefixes=NOSHDR-STR

# CHECK:       Size of this header:               64 (bytes)
# CHECK-NEXT:  Size of program headers:           56 (bytes)
# CHECK-NEXT:  Number of program headers:         6
# CHECK-NEXT:  Size of section headers:           64 (bytes)
# SHDR-NEXT:   Number of section headers:         13
# SHDR-NEXT:   Section header string table index: 11
# NOSHDR-NEXT: Number of section headers:         0
# NOSHDR-NEXT: Section header string table index: 0

# SHDR:        Section Headers:
# NOSHDR:      Symbol table for image contains 2 entries:
# NOSHDR:        _start

## _start occurs as a dynamic string table entry. There is no static string table
## entry. `nonalloc` is not in the output.
# NOSHDR-STR:      _start
# NOSHDR-STR-NOT:  _start

# RUN: not ld.lld -r -z nosectionheader %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: -r and -z nosectionheader may not be used together

.globl _start
_start:

.section nonalloc,""
.asciz "_start"
