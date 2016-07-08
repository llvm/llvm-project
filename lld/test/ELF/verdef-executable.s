# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: ld.lld %t1 -o %t
# RUN: llvm-readobj -t -V -dyn-symbols %t | FileCheck --check-prefix=EXE %s

# EXE:      Symbols [
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name:
# EXE-NEXT:      Value: 0x0
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Local
# EXE-NEXT:      Type: None (0x0)
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: Undefined (0x0)
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: _start
# EXE-NEXT:      Value: 0x11004
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: None
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: a
# EXE-NEXT:      Value: 0x11000
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: b
# EXE-NEXT:      Value: 0x11002
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: b
# EXE-NEXT:      Value: 0x11001
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: b_1
# EXE-NEXT:      Value: 0x11001
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: b_2
# EXE-NEXT:      Value: 0x11002
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:    Symbol {
# EXE-NEXT:      Name: c
# EXE-NEXT:      Value: 0x11003
# EXE-NEXT:      Size: 0
# EXE-NEXT:      Binding: Global
# EXE-NEXT:      Type: Function
# EXE-NEXT:      Other: 0
# EXE-NEXT:      Section: .text
# EXE-NEXT:    }
# EXE-NEXT:  ]
# EXE-NEXT: DynamicSymbols [
# EXE-NEXT: ]
# EXE-NEXT: Version symbols {
# EXE-NEXT: }
# EXE-NEXT: SHT_GNU_verdef {
# EXE-NEXT: }
# EXE-NEXT: SHT_GNU_verneed {
# EXE-NEXT: }

# RUN: ld.lld -pie --export-dynamic %t1 -o %t2
# RUN: llvm-readobj -t -V -dyn-symbols %t2 | \
# RUN:   FileCheck --check-prefix=EXEDYN %s

# EXEDYN:      DynamicSymbols [
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: @
# EXEDYN-NEXT:      Value: 0x0
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Local
# EXEDYN-NEXT:      Type: None
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: Undefined
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: _start@
# EXEDYN-NEXT:      Value: 0x1004
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: None
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: a@
# EXEDYN-NEXT:      Value: 0x1000
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: b@@LIBSAMPLE_2.0
# EXEDYN-NEXT:      Value: 0x1002
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: b@LIBSAMPLE_1.0
# EXEDYN-NEXT:      Value: 0x1001
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: b_1@
# EXEDYN-NEXT:      Value: 0x1001
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: b_2@
# EXEDYN-NEXT:      Value: 0x1002
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Symbol {
# EXEDYN-NEXT:      Name: c@
# EXEDYN-NEXT:      Value: 0x1003
# EXEDYN-NEXT:      Size: 0
# EXEDYN-NEXT:      Binding: Global
# EXEDYN-NEXT:      Type: Function
# EXEDYN-NEXT:      Other: 0
# EXEDYN-NEXT:      Section: .text
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:  ]
# EXEDYN-NEXT:  Version symbols {
# EXEDYN-NEXT:    Section Name: .gnu.version
# EXEDYN-NEXT:    Address: 0x288
# EXEDYN-NEXT:    Offset: 0x288
# EXEDYN-NEXT:    Link: 1
# EXEDYN-NEXT:    Symbols [
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 0
# EXEDYN-NEXT:        Name: @
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 1
# EXEDYN-NEXT:        Name: _start@
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 1
# EXEDYN-NEXT:        Name: a@
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 2
# EXEDYN-NEXT:        Name: b@@LIBSAMPLE_2.0
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 3
# EXEDYN-NEXT:        Name: b@LIBSAMPLE_1.0
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 1
# EXEDYN-NEXT:        Name: b_1@
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 1
# EXEDYN-NEXT:        Name: b_2@
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:      Symbol {
# EXEDYN-NEXT:        Version: 1
# EXEDYN-NEXT:        Name: c@
# EXEDYN-NEXT:      }
# EXEDYN-NEXT:    ]
# EXEDYN-NEXT:  }
# EXEDYN-NEXT:  SHT_GNU_verdef {
# EXEDYN-NEXT:    Definition {
# EXEDYN-NEXT:      Version: 1
# EXEDYN-NEXT:      Flags: Base
# EXEDYN-NEXT:      Index: 1
# EXEDYN-NEXT:      Hash:
# EXEDYN-NEXT:      Name:
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Definition {
# EXEDYN-NEXT:      Version: 1
# EXEDYN-NEXT:      Flags: 0x0
# EXEDYN-NEXT:      Index: 2
# EXEDYN-NEXT:      Hash: 98456416
# EXEDYN-NEXT:      Name: LIBSAMPLE_2.0
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:    Definition {
# EXEDYN-NEXT:      Version: 1
# EXEDYN-NEXT:      Flags: 0x0
# EXEDYN-NEXT:      Index: 3
# EXEDYN-NEXT:      Hash: 98457184
# EXEDYN-NEXT:      Name: LIBSAMPLE_1.0
# EXEDYN-NEXT:    }
# EXEDYN-NEXT:  }
# EXEDYN-NEXT:  SHT_GNU_verneed {
# EXEDYN-NEXT:  }

## Check when linking executable that error produced if version script
## was specified and there are symbols with version in name that
## is absent in script.
# RUN: echo    "VERSION { \
# RUN:          global: foo;   \
# RUN:          local: *; }; " > %t.script
# RUN: not ld.lld --version-script %t.script %t1 -o %t3 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR %s
# ERR: symbol b@LIBSAMPLE_1.0 has undefined version LIBSAMPLE_1.0

b@LIBSAMPLE_1.0 = b_1
b@@LIBSAMPLE_2.0 = b_2

.globl a
.type  a,@function
a:
retq

.globl b_1
.type  b_1,@function
b_1:
retq

.globl b_2
.type  b_2,@function
b_2:
retq

.globl c
.type  c,@function
c:
retq

.globl _start
_start:
 nop
