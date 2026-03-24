# REQUIRES: wasm
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/tag-export.s -o %t.tag.o
# RUN: wasm-ld -shared --experimental-pic -o %t.tag.so %t.tag.o

## Test that wasm-ld can resolve tags exported by shared libraries.
## See https://github.com/llvm/llvm-project/issues/188120

# RUN: wasm-ld --experimental-pic -pie -o %t.wasm %t.o %t.tag.so
# RUN: obj2yaml %t.wasm | FileCheck %s


  .tagtype __cpp_exception i32

  .globl _start
_start:
  .functype _start () -> ()
  i32.const 0
  throw __cpp_exception
  end_function


# CHECK:            Field:           __cpp_exception
# CHECK-NEXT:       Kind:            TAG
# CHECK-NEXT:       SigIndex:        0
