## Test that weak undefined symbols do not fetch members from archive files.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hello.s -o %t.hello.o
# RUN: rm -f %t.a
# RUN: llvm-ar rcs %t.a %t.ret32.o %t.hello.o

# RUN: wasm-ld %t.o %t.a -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

## Also test with the library symbols being read first
# RUN: wasm-ld %t.a %t.o -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s

# RUN: wasm-ld -u hello_str %t.o %t.a -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s -check-prefix=CHECK-DATA

# Weak external function symbol
.weak ret32
.functype ret32 () -> (i32)

# Weak external data symbol
.weak hello_str

.globl  _start
_start:
  .functype _start () -> ()
  block
  i32.const hello_str
  i32.eqz
  br_if 0
  call ret32
  drop
  end_block
  end_function

# Ensure we have no data section.  If we do, would mean that hello_str was
# pulled out of the library.
# CHECK-NOT:  Type:            DATA
# CHECK-DATA: Type:            DATA

# CHECK: Name: 'undefined_weak:ret32'
# CHECK-NOT: Name: ret32
