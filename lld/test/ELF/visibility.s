# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/visibility.s -o %t2.o
# RUN: ld.lld -shared %t.o %t2.o -o %t.so
# RUN: llvm-readelf -s %t.so | FileCheck %s

## Check the most constraining visibility attribute is propagated to the symbol tables.

# CHECK:      Symbol table '.dynsym' contains 3 entries:
# CHECK:      GLOBAL DEFAULT   6 default
# CHECK-NEXT: GLOBAL PROTECTED 6 protected

# CHECK:      Symbol table '.symtab' contains 7 entries:
# CHECK:      LOCAL  HIDDEN    6 hidden
# CHECK-NEXT: LOCAL  INTERNAL  6 internal
# CHECK-NEXT: LOCAL  HIDDEN    6 protected_with_hidden
# CHECK:      GLOBAL DEFAULT   6 default
# CHECK-NEXT: GLOBAL PROTECTED 6 protected

.global default
default:

.global protected
protected:

.global hidden
hidden:

.global internal
internal:

.global protected_with_hidden
.protected
protected_with_hidden:
