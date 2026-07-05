# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -o %t1.wasm %t.o
# RUN: obj2yaml %t1.wasm | FileCheck %s

.globl f1
.import_module f1, env
.import_name f1, f
.functype f1 () -> (i32)

# Same import module/name/type as `f1`, should be de-duped.
.globl f2
.import_module f2, env
.import_name f2, f
.functype f2 () -> (i32)

# Same import module/name, but different type. Should not be de-duped.
.globl f3
.import_module f3, env
.import_name f3, f
.functype f3 () -> (f32)

.globl _start
_start:
  .functype _start () -> ()
  call f1
  drop
  call f2
  drop
  call f3
  drop
  end_function


# CHECK:        - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - F32
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           f
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        0
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           f
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        1
# CHECK-NEXT:   - Type:
