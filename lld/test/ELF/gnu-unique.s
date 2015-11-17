// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: ld.lld2 %t -shared -o %tout.so
// RUN: llvm-readobj -dyn-symbols %tout.so | FileCheck %s
// REQUIRES: x86

// Check that STB_GNU_UNIQUE is treated as a global and ends up in the dynamic
// symbol table as STB_GNU_UNIQUE.

.global _start
.text
_start:

.data
.type symb, @gnu_unique_object
symb:

# CHECK:        Name: symb@
# CHECK-NEXT:   Value:
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Unique
# CHECK-NEXT:   Type: Object
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: .data
# CHECK-NEXT: }
