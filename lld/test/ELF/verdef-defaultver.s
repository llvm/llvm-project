# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/verdef-defaultver.s -o %t1
# RUN: echo "V1 { global: a; local: *; };" > %t.script
# RUN: echo "V2 { global: b; c; } V1;" >> %t.script
# RUN: ld.lld --hash-style=sysv -shared -soname shared %t1 --version-script %t.script -o %t.so
# RUN: llvm-readobj -V --dyn-syms %t.so | FileCheck --check-prefix=DSO %s

# DSO:      DynamicSymbols [
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name:
# DSO-NEXT:      Value: 0x0
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Local
# DSO-NEXT:      Type: None
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: Undefined
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: a@@V1
# DSO-NEXT:      Value: 0x12E8
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: b@@V2
# DSO-NEXT:      Value: 0x12EA
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: b@V1
# DSO-NEXT:      Value: 0x12E9
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: c@@V2
# DSO-NEXT:      Value: 0x12EB
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:  ]
# DSO-NEXT:  VersionSymbols [
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Version: 0
# DSO-NEXT:      Name:
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Version: 2
# DSO-NEXT:      Name: a@@V1
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Version: 3
# DSO-NEXT:      Name: b@@V2
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Version: 2
# DSO-NEXT:      Name: b@V1
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Version: 3
# DSO-NEXT:      Name: c@@V2
# DSO-NEXT:    }
# DSO-NEXT:  ]
# DSO-NEXT:  VersionDefinitions [
# DSO-NEXT:    Definition {
# DSO-NEXT:      Version: 1
# DSO-NEXT:      Flags [ (0x1)
# DSO-NEXT:        Base (0x1)
# DSO-NEXT:      ]
# DSO-NEXT:      Index: 1
# DSO-NEXT:      Hash: 127830196
# DSO-NEXT:      Name: shared
# DSO-NEXT:      Predecessors: []
# DSO-NEXT:    }
# DSO-NEXT:    Definition {
# DSO-NEXT:      Version: 1
# DSO-NEXT:      Flags [ (0x0)
# DSO-NEXT:      ]
# DSO-NEXT:      Index: 2
# DSO-NEXT:      Hash: 1425
# DSO-NEXT:      Name: V1
# DSO-NEXT:      Predecessors: []
# DSO-NEXT:    }
# DSO-NEXT:    Definition {
# DSO-NEXT:      Version: 1
# DSO-NEXT:      Flags [ (0x0)
# DSO-NEXT:      ]
# DSO-NEXT:      Index: 3
# DSO-NEXT:      Hash: 1426
# DSO-NEXT:      Name: V2
# DSO-NEXT:      Predecessors: []
# DSO-NEXT:    }
# DSO-NEXT:  ]

## Check that we can link against DSO produced.
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t2
# RUN: ld.lld --hash-style=sysv %t2 %t.so -o %t3
# RUN: llvm-readobj -V --dyn-syms %t3 | FileCheck --check-prefix=EXE %s

# EXE:      DynamicSymbols [
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name:
# EXE-NEXT:      Value: 0x0
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Local
# EXE-NEXT:      Type: None
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: Undefined
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: a@V1
# EXE-NEXT:      Value: 0x201340
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: Undefined
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: b@V2
# EXE-NEXT:      Value: 0x201350
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: Undefined
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: c@V2
# EXE-NEXT:      Value: 0x201360
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: Undefined
# EXE-NEXT:    }
# EXE-NEXT:  ]
# EXE-NEXT:  VersionSymbols [
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Version: 0
# EXE-NEXT:      Name:
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Version: 2
# EXE-NEXT:      Name: a@V1
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Version: 3
# EXE-NEXT:      Name: b@V2
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Version: 3
# EXE-NEXT:      Name: c@V2
# EXE-NEXT:    }
# EXE-NEXT:  ]
# EXE-NEXT:  VersionDefinitions [
# EXE-NEXT:  ]
# EXE-NEXT:  VersionRequirements [
# EXE-NEXT:    Dependency {
# EXE-NEXT:      Version: 1
# EXE-NEXT:      Count: 2
# EXE-NEXT:      FileName: shared
# EXE-NEXT:      Entries [
# EXE-NEXT:        Entry {
# EXE-NEXT:          Hash: 1425
# EXE-NEXT:          Flags [ (0x0)
# EXE-NEXT:          ]
# EXE-NEXT:          Index: 2
# EXE-NEXT:          Name: V1
# EXE-NEXT:        }
# EXE-NEXT:        Entry {
# EXE-NEXT:          Hash: 1426
# EXE-NEXT:          Flags [ (0x0)
# EXE-NEXT:          ]
# EXE-NEXT:          Index: 3
# EXE-NEXT:          Name: V2
# EXE-NEXT:        }
# EXE-NEXT:      ]
# EXE-NEXT:    }
# EXE-NEXT:  ]

.globl _start
_start:
  .long a - .
  .long b - .
  .long c - .
