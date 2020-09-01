# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-readobj -symbols %t | FileCheck %s

# CHECK:      Symbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _main
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __text (0x1)
# CHECK-NEXT:     RefType:
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value:
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: bar
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __text (0x1)
# CHECK-NEXT:     RefType:
# CHECK-NEXT:     Flags [ (0x80)
# CHECK-NEXT:       WeakDef (0x80)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value:
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: foo
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __data
# CHECK-NEXT:     RefType:
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value:
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.data
.global foo
foo:
  .asciz "Hello world!\n"

.text
.weak_definition bar
.global bar
.global _main

_main:
  mov $0, %rax
  ret

bar:
  mov $2, %rax
  ret
