# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname shared -o %t2.so

# RUN: echo "{ global: foo1; foo3; local: *; };" > %t.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --version-script %t.script -shared %t.o %t2.so -o %t.so
# RUN: llvm-readobj -dyn-symbols %t.so | FileCheck --check-prefix=DSO %s

# RUN: echo "{ local: *; };" > %t3.script
# RUN: ld.lld --version-script %t3.script -shared %t.o %t2.so -o %t3.so
# RUN: llvm-readobj -dyn-symbols %t3.so | FileCheck --check-prefix=DSO2 %s

# --version-script filters --dynamic-list.
# RUN: echo "{ foo1; foo2; };" > %t.list
# RUN: ld.lld --version-script %t.script --dynamic-list %t.list %t.o %t2.so -o %t
# RUN: llvm-readobj -dyn-symbols %t | FileCheck --check-prefix=EXE %s

# DSO:      DynamicSymbols [
# DSO-NEXT:   Symbol {
# DSO-NEXT:     Name: @ (0)
# DSO-NEXT:     Value: 0x0
# DSO-NEXT:     Size: 0
# DSO-NEXT:     Binding: Local (0x0)
# DSO-NEXT:     Type: None (0x0)
# DSO-NEXT:     Other: 0
# DSO-NEXT:     Section: Undefined (0x0)
# DSO-NEXT:   }
# DSO-NEXT:   Symbol {
# DSO-NEXT:     Name: bar@ (1)
# DSO-NEXT:     Value: 0x0
# DSO-NEXT:     Size: 0
# DSO-NEXT:     Binding: Global (0x1)
# DSO-NEXT:     Type: Function (0x2)
# DSO-NEXT:     Other: 0
# DSO-NEXT:     Section: Undefined (0x0)
# DSO-NEXT:   }
# DSO-NEXT:   Symbol {
# DSO-NEXT:     Name: foo1@ (5)
# DSO-NEXT:     Value: 0x1000
# DSO-NEXT:     Size: 0
# DSO-NEXT:     Binding: Global (0x1)
# DSO-NEXT:     Type: None (0x0)
# DSO-NEXT:     Other: 0
# DSO-NEXT:     Section: .text (0x5)
# DSO-NEXT:   }
# DSO-NEXT:   Symbol {
# DSO-NEXT:     Name: foo3@ (10)
# DSO-NEXT:     Value: 0x1007
# DSO-NEXT:     Size: 0
# DSO-NEXT:     Binding: Global (0x1)
# DSO-NEXT:     Type: None (0x0)
# DSO-NEXT:     Other: 0
# DSO-NEXT:     Section: .text (0x5)
# DSO-NEXT:   }
# DSO-NEXT: ]

# DSO2:      DynamicSymbols [
# DSO2-NEXT:   Symbol {
# DSO2-NEXT:     Name: @ (0)
# DSO2-NEXT:     Value: 0x0
# DSO2-NEXT:     Size: 0
# DSO2-NEXT:     Binding: Local (0x0)
# DSO2-NEXT:     Type: None (0x0)
# DSO2-NEXT:     Other: 0
# DSO2-NEXT:     Section: Undefined (0x0)
# DSO2-NEXT:   }
# DSO2-NEXT:   Symbol {
# DSO2-NEXT:     Name: bar@ (1)
# DSO2-NEXT:     Value: 0x0
# DSO2-NEXT:     Size: 0
# DSO2-NEXT:     Binding: Global (0x1)
# DSO2-NEXT:     Type: Function (0x2)
# DSO2-NEXT:     Other: 0
# DSO2-NEXT:     Section: Undefined (0x0)
# DSO2-NEXT:   }
# DSO2-NEXT: ]

# EXE:      DynamicSymbols [
# EXE-NEXT:   Symbol {
# EXE-NEXT:     Name: @ (0)
# EXE-NEXT:     Value: 0x0
# EXE-NEXT:     Size: 0
# EXE-NEXT:     Binding: Local (0x0)
# EXE-NEXT:     Type: None (0x0)
# EXE-NEXT:     Other: 0
# EXE-NEXT:     Section: Undefined (0x0)
# EXE-NEXT:   }
# EXE-NEXT:   Symbol {
# EXE-NEXT:     Name: bar@ (1)
# EXE-NEXT:     Value: 0x0
# EXE-NEXT:     Size: 0
# EXE-NEXT:     Binding: Global (0x1)
# EXE-NEXT:     Type: Function (0x2)
# EXE-NEXT:     Other: 0
# EXE-NEXT:     Section: Undefined (0x0)
# EXE-NEXT:   }
# EXE-NEXT:   Symbol {
# EXE-NEXT:     Name: foo1@ (5)
# EXE-NEXT:     Value: 0x11000
# EXE-NEXT:     Size: 0
# EXE-NEXT:     Binding: Global (0x1)
# EXE-NEXT:     Type: None (0x0)
# EXE-NEXT:     Other: 0
# EXE-NEXT:     Section: .text (0x5)
# EXE-NEXT:   }
# EXE-NEXT: ]

.globl foo1
foo1:
  call bar@PLT
  ret

.globl foo2
foo2:
  ret

.globl foo3
foo3:
  call foo2@PLT
  ret

.globl _start
_start:
  ret
