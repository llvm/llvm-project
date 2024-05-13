# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl _start
.globl fn_i32
.globl fn_i32_i32
.globl fn_i32_i64
.globl fn_i64_f64_i32_f32


fn_i32:
  .functype fn_i32 () -> (i32)
  i32.const 1
  end_function

fn_i32_i32:
  .functype fn_i32_i32 () -> (i32, i32)
  i32.const 1
  i32.const 1
  end_function

fn_i32_i64:
  .functype fn_i32_i64 () -> (i32, i64)
  i32.const 1
  i64.const 1
  end_function

fn_i64_f64_i32_f32:
  .functype fn_i64_f64_i32_f32 () -> (i64, f64, i32, f32)
  i64.const 1
  f64.const 1.0
  i32.const 1
  f32.const 1.0
  end_function

_start:
  .functype _start () -> ()
  call fn_i32
  drop
  call fn_i32_i32
  drop
  drop
  call fn_i32_i64
  drop
  drop
  call fn_i64_f64_i32_f32
  drop
  drop
  drop
  drop
  end_function


# CHECK:       - Type:            TYPE
# CHECK-NEXT:    Signatures:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:     []
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:           - I32
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:           - I64
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:           - F64
# CHECK-NEXT:           - I32
# CHECK-NEXT:           - F32
