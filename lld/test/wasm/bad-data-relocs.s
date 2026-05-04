## Certain relocations types are not supported by runtime relocation code
## generated in `-shared/`-pie` binaries.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld -pie --experimental-pic %t.o -o %t.wasm 2>&1 | FileCheck %s

# CHECK: wasm-ld: error: invalid runtime relocation type in data section: R_WASM_FUNCTION_INDEX_I32

foo:
  .functype foo (i32) -> ()
  end_function

.globl _start
_start:
  .functype _start () -> ()
  i32.const bar@GOT
  call foo@GOT
  end_function

# data section containing relocation type that is not valid in a data section
.section .data,"",@
.globl bar
bar:
  .int32 0
  .size  bar, 4

.reloc bar, R_WASM_FUNCTION_INDEX_I32, foo
