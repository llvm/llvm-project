# RUN: llvm-mc -mattr=+reference-types -triple=wasm32-unknown-unknown -filetype=obj -o %t.o %s
# RUN: wasm-ld --allow-undefined -o %t1.wasm %t.o
# RUN: obj2yaml %t1.wasm | FileCheck %s

.tabletype t1, funcref
.import_module t1, env
.import_name t1, t
.globl t1

# Same import module/name/type as `t1`, should be de-duped.
.tabletype t2, funcref
.import_module t2, env
.import_name t2, t
.globl t2

# Imported as an externref instead of funcref, so should not be de-duped.
.tabletype t3, externref
.import_module t3, env
.import_name t3, t
.globl t3

.globl _start
_start:
  .functype _start () -> ()

  # Read from `t1`
  i32.const 0
  table.get t1
  drop

  # Read from `t2`
  i32.const 0
  table.get t2
  drop

  # Read from `t3`
  i32.const 0
  table.get t3
  drop

  end_function

## XXX: the second imported table has index 1, not 0. I've verified by hand
## (with `wasm2wat`) that the resulting Wasm file is correct: `t3` does end up
## at index 1 and our `table.get` instructions are using the proper table index
## immediates. This is also asserted (less legibly) in the hexdump of the code
## body below. It looks like there's a bug in how `obj2yaml` disassembles
## multiple table imports.

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           t
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Minimum:         0x0
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           t
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        EXTERNREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Minimum:         0x0
# CHECK-NEXT:   - Type:

# CHECK:        - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41002580808080001A41002580808080001A41002581808080001A0B
# CHECK-NEXT:   - Type:
