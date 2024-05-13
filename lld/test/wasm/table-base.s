# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o

# RUN: wasm-ld --export=__table_base -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-DEFAULT

# RUN: wasm-ld --table-base=100 --export=__table_base -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-100

.globl _start
_start:
  .functype _start () -> ()
  i32.const _start
  drop
  end_function

# CHECK-DEFAULT:       - Type:            TABLE
# CHECK-DEFAULT-NEXT:    Tables:
# CHECK-DEFAULT-NEXT:      - Index:           0
# CHECK-DEFAULT-NEXT:        ElemType:        FUNCREF
# CHECK-DEFAULT-NEXT:        Limits:
# CHECK-DEFAULT-NEXT:          Flags:           [ HAS_MAX ]
# CHECK-DEFAULT-NEXT:          Minimum:         0x2
# CHECK-DEFAULT-NEXT:          Maximum:         0x2

# CHECK-DEFAULT:       - Type:            GLOBAL
# CHECK-DEFAULT-NEXT:    Globals:
# CHECK-DEFAULT-NEXT:      - Index:           0
# CHECK-DEFAULT-NEXT:        Type:            I32
# CHECK-DEFAULT-NEXT:        Mutable:         true
# CHECK-DEFAULT-NEXT:        InitExpr:
# CHECK-DEFAULT-NEXT:          Opcode:          I32_CONST
# CHECK-DEFAULT-NEXT:          Value:           66560
# CHECK-DEFAULT-NEXT:      - Index:           1
# CHECK-DEFAULT-NEXT:        Type:            I32
# CHECK-DEFAULT-NEXT:        Mutable:         false
# CHECK-DEFAULT-NEXT:        InitExpr:
# CHECK-DEFAULT-NEXT:          Opcode:          I32_CONST
# CHECK-DEFAULT-NEXT:          Value:           1

# CHECK-DEFAULT:       - Type:            EXPORT
# CHECK-DEFAULT:           - Name:            __table_base
# CHECK-DEFAULT-NEXT:        Kind:            GLOBAL
# CHECK-DEFAULT-NEXT:        Index:           1

# CHECK-100:       - Type:            TABLE
# CHECK-100-NEXT:    Tables:
# CHECK-100-NEXT:      - Index:           0
# CHECK-100-NEXT:        ElemType:        FUNCREF
# CHECK-100-NEXT:        Limits:
# CHECK-100-NEXT:          Flags:           [ HAS_MAX ]
# CHECK-100-NEXT:          Minimum:         0x65
# CHECK-100-NEXT:          Maximum:         0x65

# CHECK-100:       - Type:            GLOBAL
# CHECK-100-NEXT:    Globals:
# CHECK-100-NEXT:      - Index:           0
# CHECK-100-NEXT:        Type:            I32
# CHECK-100-NEXT:        Mutable:         true
# CHECK-100-NEXT:        InitExpr:
# CHECK-100-NEXT:          Opcode:          I32_CONST
# CHECK-100-NEXT:          Value:           66560
# CHECK-100-NEXT:      - Index:           1
# CHECK-100-NEXT:        Type:            I32
# CHECK-100-NEXT:        Mutable:         false
# CHECK-100-NEXT:        InitExpr:
# CHECK-100-NEXT:          Opcode:          I32_CONST
# CHECK-100-NEXT:          Value:           100

# CHECK-100:       - Type:            EXPORT
# CHECK-100:           - Name:            __table_base
# CHECK-100-NEXT:        Kind:            GLOBAL
# CHECK-100-NEXT:        Index:           1
