# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl _start
.globl read_global
.globl write_global

.globaltype foo_global, i32
.globaltype bar_global, f32
.globaltype immutable_global, i32, immutable

read_global:
  .functype read_global () -> (i32)
  global.get immutable_global
  end_function

write_global:
  .functype write_global (i32) -> ()
  local.get 0
  global.set foo_global
  f32.const 1.0
  global.set bar_global
  end_function

_start:
  .functype _start () -> ()
  i32.const 1
  call write_global
  call read_global
  drop
  end_function

foo_global:
bar_global:
immutable_global:

# CHECK:       - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           66560
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Type:            F32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          F32_CONST
# CHECK-NEXT:          Value:           0
