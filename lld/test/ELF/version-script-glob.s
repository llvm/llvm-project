# REQUIRES: x86

# RUN: echo "{ global: foo*; bar*; local: *; };" > %t.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -shared --version-script %t.script %t.o -o %t.so
# RUN: llvm-readobj -dyn-symbols %t.so | FileCheck %s

        .globl foo1
foo1:

        .globl bar1
bar1:

        .globl zed1
zed1:

# CHECK:      DynamicSymbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name:
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: bar1
# CHECK-NEXT:     Value: 0x1000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: foo1
# CHECK-NEXT:     Value: 0x1000
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK-NEXT: ]
