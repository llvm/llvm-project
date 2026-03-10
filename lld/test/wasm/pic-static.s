# Test that PIC code can be linked into static binaries.
# In this case the GOT entries will end up as internalized wasm globals with
# fixed values.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --allow-undefined --export-all -o %t.wasm %t.o %t.ret32.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globaltype __memory_base, i32, immutable
.globaltype __table_base, i32, immutable
.weak missing_float

.functype ret32 (f32) -> (i32)
.functype missing_function (f32) -> (i32)

.globl getaddr_external
getaddr_external:
.functype getaddr_external () -> (i32)
  global.get ret32@GOT
  end_function

.globl getaddr_missing_function
getaddr_missing_function:
  .functype getaddr_missing_function () -> (i32)
  global.get missing_function@GOT
  end_function

.globl getaddr_hidden
getaddr_hidden:
  .functype getaddr_hidden () -> (i32)
  global.get  __table_base
  i32.const hidden_func@TBREL
  i32.add
  end_function

.globl getaddr_missing_float
getaddr_missing_float:
  .functype getaddr_missing_float () -> (i32)
  global.get missing_float@GOT
  end_function

.hidden hidden_func
.globl hidden_func
hidden_func:
  .functype hidden_func () -> (i32)
  i32.const 1
  end_function

.globl _start
_start:
  .functype _start () -> ()
  .local    i32, i32
  global.get global_float@GOT
  f32.load 0
  global.get ret32_ptr@GOT
  i32.load 0
  call_indirect (f32) -> (i32)
  drop
  global.get __memory_base
  local.set 0
  call  getaddr_external
  local.set 1
  local.get 0
  i32.const hidden_float@MBREL
  i32.add
  f32.load 0
  local.get 1
  call_indirect (f32) -> (i32)
  drop
  call getaddr_hidden
  call_indirect () -> (i32)
  drop
  end_function

.section .data.global_float,"",@
.globl global_float
global_float:
  .int32  0x3f800000
  .size global_float, 4

.hidden hidden_float
.section .data.hidden_float,"",@
.globl hidden_float
hidden_float:
  .int32 0x40000000
  .size hidden_float, 4

.section .data.ret32_ptr,"",@
.globl ret32_ptr
ret32_ptr:
  .int32 ret32
  .size ret32_ptr, 4

# CHECK:        - Type:            GLOBAL
# CHECK-NEXT:     Globals:

# __stack_pointer
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536

# __memory_base
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0

# __table_base
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1

# __tls_base
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0

# GOT.func.internal.ret32
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value: 1

# GOT.func.missing_function
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           2

# GOT.mem.missing_float
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0

# GOT.mem.global_float
# CHECK-NEXT:       - Index:           7
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536

# GOT.mem.ret32_ptr
# CHECK-NEXT:       - Index:           8
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65544

# CHECK:          GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            __memory_base
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            __table_base
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            __tls_base
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Name:            GOT.func.internal.ret32
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Name:            GOT.func.internal.missing_function
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Name:            GOT.data.internal.missing_float
# CHECK-NEXT:       - Index:           7
# CHECK-NEXT:         Name:            GOT.data.internal.global_float
# CHECK-NEXT:       - Index:           8
# CHECK-NEXT:         Name:            GOT.data.internal.ret32_ptr
# CHECK-NEXT:     DataSegmentNames:
