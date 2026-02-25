## Test linking of symbols with dollar signs in their names.
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/dollar.o %t/dollar.s
# RUN: obj2yaml %t/dollar.o | FileCheck %s --check-prefix=CHECK-OBJ-DOLLAR
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: obj2yaml %t/main.o | FileCheck %s --check-prefix=CHECK-OBJ-MAIN
# RUN: wasm-ld %t/dollar.o %t/main.o -o %t/out.wasm 2>&1
# RUN: obj2yaml %t/out.wasm | FileCheck %s --check-prefix=CHECK-OUT

# CHECK-OBJ-DOLLAR:     SymbolTable:
# CHECK-OBJ-DOLLAR-NEXT:      - Index:           0
# CHECK-OBJ-DOLLAR-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-DOLLAR-NEXT:        Name:            '$dollar_func'
# CHECK-OBJ-DOLLAR-NEXT:        Flags:           [  ]
# CHECK-OBJ-DOLLAR-NEXT:        Function:        0
# CHECK-OBJ-DOLLAR-NEXT:      - Index:           1
# CHECK-OBJ-DOLLAR-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-DOLLAR-NEXT:        Name:            'mid$dollar_func'
# CHECK-OBJ-DOLLAR-NEXT:        Flags:           [  ]
# CHECK-OBJ-DOLLAR-NEXT:        Function:        1
# CHECK-OBJ-DOLLAR-NEXT:      - Index:           2
# CHECK-OBJ-DOLLAR-NEXT:        Kind:            DATA
# CHECK-OBJ-DOLLAR-NEXT:        Name:            '$dollar_global'
# CHECK-OBJ-DOLLAR-NEXT:        Flags:           [  ]
# CHECK-OBJ-DOLLAR-NEXT:        Segment:         0
# CHECK-OBJ-DOLLAR-NEXT:        Size:            4
# CHECK-OBJ-DOLLAR-NEXT:      - Index:           3
# CHECK-OBJ-DOLLAR-NEXT:        Kind:            DATA
# CHECK-OBJ-DOLLAR-NEXT:        Name:            'mid$dollar_global'
# CHECK-OBJ-DOLLAR-NEXT:        Flags:           [  ]
# CHECK-OBJ-DOLLAR-NEXT:        Segment:         1
# CHECK-OBJ-DOLLAR-NEXT:        Size:            4

# CHECK-OBJ-MAIN:    SymbolTable:
# CHECK-OBJ-MAIN-NEXT:      - Index:           0
# CHECK-OBJ-MAIN-NEXT:        Kind:            DATA
# CHECK-OBJ-MAIN-NEXT:        Name:            '$dollar_global'
# CHECK-OBJ-MAIN-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-MAIN-NEXT:      - Index:           1
# CHECK-OBJ-MAIN-NEXT:        Kind:            DATA
# CHECK-OBJ-MAIN-NEXT:        Name:            'mid$dollar_global'
# CHECK-OBJ-MAIN-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-MAIN-NEXT:      - Index:           2
# CHECK-OBJ-MAIN-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-MAIN-NEXT:        Name:            _start
# CHECK-OBJ-MAIN-NEXT:        Flags:           [  ]
# CHECK-OBJ-MAIN-NEXT:        Function:        2
# CHECK-OBJ-MAIN-NEXT:      - Index:           3
# CHECK-OBJ-MAIN-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-MAIN-NEXT:        Name:            '$dollar_func'
# CHECK-OBJ-MAIN-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-MAIN-NEXT:        Function:        0
# CHECK-OBJ-MAIN-NEXT:      - Index:           4
# CHECK-OBJ-MAIN-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-MAIN-NEXT:        Name:            'mid$dollar_func'
# CHECK-OBJ-MAIN-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-MAIN-NEXT:        Function:        1

# CHECK-OUT:        - Type:            CUSTOM
# CHECK-OUT-NEXT:        Name:            name
# CHECK-OUT-NEXT:        FunctionNames:
# CHECK-OUT-NEXT:          - Index:           0
# CHECK-OUT-NEXT:            Name:            '$dollar_func'
# CHECK-OUT-NEXT:          - Index:           1
# CHECK-OUT-NEXT:            Name:            'mid$dollar_func'
# CHECK-OUT-NEXT:          - Index:           2

#--- main.s

.functype $dollar_func () -> ()
.functype mid$dollar_func () -> ()

.globl $dollar_global
.globl mid$dollar_global

.globl _start
_start:
  .functype _start () -> ()
  call $dollar_func
  call mid$dollar_func
  i32.const $dollar_global
  drop
  i32.const mid$dollar_global
  drop
  end_function

#--- dollar.s

.globl $dollar_func
$dollar_func:
  .functype $dollar_func () -> ()
  end_function

.globl mid$dollar_func
mid$dollar_func:
  .functype mid$dollar_func () -> ()
  end_function

.globl $dollar_global
.section  ".data.$dollar_global","",@
$dollar_global:
  .int32 42
  .size $dollar_global, 4

.globl mid$dollar_global
.section  ".data.mid$dollar_global","",@
mid$dollar_global:
  .int32 43
  .size mid$dollar_global, 4
