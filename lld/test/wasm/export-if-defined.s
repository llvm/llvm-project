# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --export-if-defined=foo -o %t1.wasm %t.o
# RUN: obj2yaml %t1.wasm | FileCheck %s

# RUN: wasm-ld --export-if-defined=bar -o %t2.wasm %t.o
# RUN: obj2yaml %t2.wasm | FileCheck %s --check-prefixes=MISSING

.globl foo
foo:
  .functype foo () -> ()
  end_function

.globl _start
_start:
  .functype _start () -> ()
  end_function

#      CHECK:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            foo
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1

#      MISSING:   - Type:            EXPORT
# MISSING-NEXT:     Exports:
# MISSING-NEXT:       - Name:            memory
# MISSING-NEXT:         Kind:            MEMORY
# MISSING-NEXT:         Index:           0
# MISSING-NEXT:       - Name:            _start
# MISSING-NEXT:         Kind:            FUNCTION
# MISSING-NEXT:         Index:           0
