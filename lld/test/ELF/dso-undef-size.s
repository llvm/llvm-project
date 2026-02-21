# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %S/Inputs/dso-undef-size.s -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t2.o
# RUN: ld.lld -shared %t2.o %t1.so -o %t2.so
# RUN: llvm-readobj --symbols --dyn-syms %t2.so | FileCheck %s

# CHECK: Symbols [
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name:  (0)
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Local (0x0)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Undefined (0x0)
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: _DYNAMIC (5)
# skip actual value here:
# CHECK-NEXT:    Value:
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Local (0x0)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other [ (0x2)
# CHECK-NEXT:      STV_HIDDEN (0x2)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Section: .dynamic (0x6)
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: foo (1)
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global (0x1)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Undefined (0x0)
# CHECK-NEXT:  }
# CHECK-NEXT:]
# CHECK-NEXT:DynamicSymbols [
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name:  (0)
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Local (0x0)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Undefined (0x0)
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: foo (1)
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global (0x1)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Undefined (0x0)
# CHECK-NEXT:  }
# CHECK-NEXT:]

.text
.global foo
