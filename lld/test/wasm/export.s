# Test in default mode
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld --export=missing -o %t.wasm %t.o 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
# RUN: wasm-ld --export=hidden_function -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

# Now test in Emscripten mode
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: not wasm-ld --export=missing -o %t.wasm %t.o 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
# RUN: wasm-ld --export=hidden_function -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefixes=CHECK,EMSCRIPTEN

# Not exported by default, but forced via commandline
  .hidden hidden_function
  .globl hidden_function
hidden_function:
  .functype hidden_function () -> (i32)
  i32.const 0
  end_function

# Not exported by default
  .globl default_function
default_function:
  .functype default_function () -> (i32)
  i32.const 0
  end_function

# Exported in emscripten mode which treats .no_dead_strip as a signal to export
  .no_dead_strip used_function
  .globl used_function
used_function:
  .functype used_function () -> (i32)
  i32.const 0
  end_function

# Exported by default
.globl _start
_start:
  .functype _start () -> ()
  end_function

# CHECK-ERROR: error: symbol exported via --export not found: missing

# CHECK-NOT: - Name: default_function

# CHECK:        - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            hidden_function
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# EMSCRIPTEN-NEXT:  - Name:            used_function
# EMSCRIPTEN-NEXT:    Kind:            FUNCTION
# EMSCRIPTEN-NEXT:    Index:           1
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:   - Type:            CODE
