# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic --unresolved-symbols=import-dynamic %t.o -o %t.wasm

.functype foo () -> ()
.functype bar () -> ()

.globl _start
_start:
  .functype _start () -> ()
  call foo
  call bar
  end_function

.section .custom_section.target_features,"",@
.int8 1
.int8 43
.int8 15
.ascii "compact-imports"

# Neither llvm-readobj nor obj2yaml currently report compact imports differently
# so the check here is just for the size of the import section.  The Size here
# is larger than 20 bytes without compact imports enabled.

# RUN: llvm-readobj --sections %t.wasm | FileCheck %s

# CHECK:         Type: IMPORT (0x2)
# CHECK-NEXT:    Size: 20
