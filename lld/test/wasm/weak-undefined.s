# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -strip-all %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test that undefined weak externals (global_var) and (foo) don't cause
# link failures and resolve to zero.

.functype foo () -> (i32)

.globl  get_address_of_foo
get_address_of_foo:
  .functype get_address_of_foo () -> (i32)
  i32.const foo
  end_function

.globl  get_address_of_global_var
get_address_of_global_var:
  .functype get_address_of_global_var () -> (i32)
  i32.const global_var
  end_function

.globl  _start
_start:
  .functype _start () -> ()
  call  get_address_of_global_var
  drop
  call  get_address_of_foo
  drop
  call  foo
  drop
  end_function

.weak foo
.weak global_var


# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0, 0, 1 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x1
# CHECK-NEXT:           Maximum:         0x1
# CHECK-NEXT:   - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHECK-NEXT:       - Minimum:         0x2
# CHECK-NEXT:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66560
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            000B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            4180808080000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            4180808080000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1082808080001A1081808080001A1080808080001A0B
# CHECK-NEXT: ...
