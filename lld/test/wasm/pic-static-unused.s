# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globaltype __memory_base, i32, immutable
.globaltype __table_base, i32, immutable

.globl _start
_start:
  .functype _start () -> ()
  end_function

.section .debug_info,"",@
  .int32 __memory_base
  .int32 __table_base

## Check that relocations against unused __memory_base and __table_base
## work in non-pic mode.

# CHECK:         Name:            .debug_info
# CHECK-NEXT:    Payload:         FFFFFFFFFFFFFFFF
