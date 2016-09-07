# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "LIBSAMPLE_1.0 { \
# RUN:   global:             \
# RUN:      extern \"C++\" { \
# RUN:         \"foo(int)\"; \
# RUN:         \"zed(int)\"; \
# RUN:   };                  \
# RUN: };                    \
# RUN: LIBSAMPLE_2.0 {       \
# RUN:   global:             \
# RUN:     extern \"C++\" {  \
# RUN:       \"bar(int)\";   \
# RUN:   };                  \
# RUN: }; " > %t.script
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj -V -dyn-symbols %t.so | FileCheck --check-prefix=DSO %s

# DSO:      DynamicSymbols [
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: @
# DSO-NEXT:      Value: 0x0
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Local
# DSO-NEXT:      Type: None
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: Undefined
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3bari@@LIBSAMPLE_2.0
# DSO-NEXT:      Value: 0x1001
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3fooi@@LIBSAMPLE_1.0
# DSO-NEXT:      Value: 0x1000
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global
# DSO-NEXT:      Type: Function
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text
# DSO-NEXT:    }
# DSO-NEXT:    Symbol {
# DSO-NEXT:      Name: _Z3zedi@@LIBSAMPLE_1.0
# DSO-NEXT:      Value: 0x1002
# DSO-NEXT:      Size: 0
# DSO-NEXT:      Binding: Global (0x1)
# DSO-NEXT:      Type: Function (0x2)
# DSO-NEXT:      Other: 0
# DSO-NEXT:      Section: .text (0x6)
# DSO-NEXT:    }
# DSO-NEXT:  ]
# DSO-NEXT:  Version symbols {
# DSO-NEXT:    Section Name: .gnu.version
# DSO-NEXT:    Address: 0x228
# DSO-NEXT:    Offset: 0x228
# DSO-NEXT:    Link: 1
# DSO-NEXT:    Symbols [
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 0
# DSO-NEXT:        Name: @
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 3
# DSO-NEXT:        Name: _Z3bari@@LIBSAMPLE_2.0
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 2
# DSO-NEXT:        Name: _Z3fooi@@LIBSAMPLE_1.0
# DSO-NEXT:      }
# DSO-NEXT:      Symbol {
# DSO-NEXT:        Version: 2
# DSO-NEXT:        Name: _Z3zedi@@LIBSAMPLE_1.0
# DSO-NEXT:      }
# DSO-NEXT:    ]
# DSO-NEXT:  }

.text
.globl _Z3fooi
.type _Z3fooi,@function
_Z3fooi:
retq

.globl _Z3bari
.type _Z3bari,@function
_Z3bari:
retq

.globl _Z3zedi
.type _Z3zedi,@function
_Z3zedi:
retq
