# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -wrap nosuchsym -wrap foo -o %t.wasm %t.o
# RUN: wasm-ld -emit-relocs -wrap foo -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype __real_foo () -> (i32)
.globl foo
.globl _start
.globl __wrap_foo

foo:
  .functype foo () -> (i32)
  i32.const 1
  end_function

_start:
  .functype _start () -> ()
  call  foo
  drop
  end_function

__wrap_foo:
  .functype __wrap_foo () -> (i32)
  call  __real_foo
  end_function

# CHECK:      - Type:            CODE
# CHECK-NEXT:   Relocations:
# CHECK-NEXT:     - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:       Index:           2
# CHECK-NEXT:       Offset:
# CHECK-NEXT:     - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:       Index:           0
# CHECK-NEXT:       Offset:

# CHECK:        FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            foo
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            __wrap_foo
