# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: %lld -dylib %t/libfoo.o -o %t/libfoo.dylib
# RUN: %lld -lSystem %t/test.o %t/libfoo.dylib -o %t/test

# RUN: llvm-readobj --syms --macho-dysymtab %t/test | FileCheck %s
# CHECK:      Symbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _local
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __data
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x1{{[0-9a-f]*}}
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: __dyld_private
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __data
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x1{{[0-9a-f]*}}
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _main
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __text (0x1)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x1{{[0-9a-f]*}}
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _external_weak
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __text (0x1)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x80)
# CHECK-NEXT:       WeakDef (0x80)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x1{{[0-9a-f]*}}
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _external
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __data
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x1{{[0-9a-f]*}}
# CHECK-NEXT:   }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: __mh_execute_header (81)
# CHECK-NEXT:    Extern
# CHECK-NEXT:    Type: Abs (0x2)
# CHECK-NEXT:    Section:  (0x0)
# CHECK-NEXT:    RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:    Flags [ (0x0)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:  }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: dyld_stub_binder
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Undef (0x0)
# CHECK-NEXT:     Section:  (0x0)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x100)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: _dynamic
# CHECK-NEXT:     Extern
# CHECK-NEXT:     Type: Undef (0x0)
# CHECK-NEXT:     Section:  (0x0)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x200)
# CHECK-NEXT:       AltEntry (0x200)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Dysymtab {
# CHECK-NEXT:   ilocalsym: 0
# CHECK-NEXT:   nlocalsym: 2
# CHECK-NEXT:   iextdefsym: 2
# CHECK-NEXT:   nextdefsym: 4
# CHECK-NEXT:   iundefsym: 6
# CHECK-NEXT:   nundefsym: 2

## Verify that the first entry in the StringTable is a space, and that
## unreferenced symbols aren't emitted.
# RUN: obj2yaml %t/test | FileCheck %s --check-prefix=YAML
# YAML:      StringTable:
# YAML-NEXT: ' '
# YAML-NOT: _unreferenced

#--- libfoo.s
.globl _dynamic
_dynamic:

#--- test.s
.globl _main, _external, _external_weak, _unreferenced

.data
_external:
  .space 0
_local:
  .space 0

.text
.weak_definition _external_weak
_external_weak:
  .space 0

_main:
  callq _dynamic
  mov $0, %rax
  ret
