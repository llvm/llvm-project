# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -o %t.wasm %t.o
# RUN: llvm-readobj --file-headers %t.wasm | FileCheck %s

  .globl  _start
_start:
  .functype _start () -> ()
  end_function

# CHECK: Format: WASM
# CHECK: Arch: wasm32
# CHECK: AddressSize: 32bit
# CHECK: Version: 0x1
