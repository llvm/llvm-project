# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
#
# Should fail without mutable globals feature enabled.
# RUN: not wasm-ld --export-all %t.o -o %t.wasm 2>&1 | FileCheck -check-prefix=CHECK-ERR %s
# RUN: not wasm-ld --export=foo_global %t.o -o %t.wasm 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

# RUN: wasm-ld --extra-features=mutable-globals --export=foo_global %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Explcitly check that __stack_pointer can be exported
# RUN: wasm-ld --extra-features=mutable-globals --export=__stack_pointer %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck -check-prefix=CHECK-SP %s

# RUN: wasm-ld --extra-features=mutable-globals --export-all %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck -check-prefix=CHECK-ALL %s

.globl _start
.globl foo_global

.globaltype foo_global, i32
foo_global:

_start:
  .functype _start () -> ()
  end_function

# Add a target feature and ensure that it is preserved when --extra-features is
# used above.
.section  .custom_section.target_features,"",@
  .int8 1
  .int8 43
  .int8 7
  .ascii  "atomics"

# CHECK-ERR: mutable global exported but 'mutable-globals' feature not present in inputs: `foo_global`. Use --no-check-features to suppress

#      CHECK:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            foo_global
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        Index:           1
# CHECK-NEXT:  - Type:            CODE

#      CHECK-SP:  - Type:            EXPORT
# CHECK-SP-NEXT:    Exports:
# CHECK-SP-NEXT:      - Name:            memory
# CHECK-SP-NEXT:        Kind:            MEMORY
# CHECK-SP-NEXT:        Index:           0
# CHECK-SP-NEXT:      - Name:            __stack_pointer
# CHECK-SP-NEXT:        Kind:            GLOBAL
# CHECK-SP-NEXT:        Index:           0
# CHECK-SP-NEXT:      - Name:            _start
# CHECK-SP-NEXT:        Kind:            FUNCTION
# CHECK-SP-NEXT:        Index:           0
# CHECK-SP-NEXT:  - Type:            CODE

#      CHECK-ALL:  - Type:            EXPORT
# CHECK-ALL-NEXT:    Exports:
# CHECK-ALL-NEXT:      - Name:            memory
# CHECK-ALL-NEXT:        Kind:            MEMORY
# CHECK-ALL-NEXT:        Index:           0
# CHECK-ALL-NEXT:      - Name:            __wasm_call_ctors
# CHECK-ALL-NEXT:        Kind:            FUNCTION
# CHECK-ALL-NEXT:        Index:           0
# CHECK-ALL-NEXT:      - Name:            _start
# CHECK-ALL-NEXT:        Kind:            FUNCTION
# CHECK-ALL-NEXT:        Index:           1
# CHECK-ALL-NEXT:      - Name:            foo_global
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           1
# CHECK-ALL-NEXT:      - Name:            __dso_handle
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           2
# CHECK-ALL-NEXT:      - Name:            __data_end
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           3
# CHECK-ALL-NEXT:      - Name:            __global_base
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           4
# CHECK-ALL-NEXT:      - Name:            __heap_base
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           5
# CHECK-ALL-NEXT:      - Name:            __memory_base
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           6
# CHECK-ALL-NEXT:      - Name:            __table_base
# CHECK-ALL-NEXT:        Kind:            GLOBAL
# CHECK-ALL-NEXT:        Index:           7
# CHECK-ALL-NEXT:  - Type:            CODE

# CHECK-ALL:         Name:            target_features
# CHECK-ALL-NEXT:    Features:
# CHECK-ALL-NEXT:      - Prefix:          USED
# CHECK-ALL-NEXT:        Name:            atomics
# CHECK-ALL-NEXT:      - Prefix:          USED
# CHECK-ALL-NEXT:        Name:            mutable-globals
# CHECK-ALL-NEXT: ...

