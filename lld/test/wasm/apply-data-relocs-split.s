## The runtime relocation code that goes into __wasm_apply_data_relocs is split
## across several functions when it would exceed the maximum function body
## size.  --max-function-body-size= lowers the limit so that a small input is
## enough to see it happen.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

## Each of the five relocations below turns into 10 bytes of code; with the
## locals declaration and the final END they fit exactly in a 52-byte body.

## Default limit, and a limit they fit in exactly: one function, no helpers.
# RUN: wasm-ld -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=ONE
# RUN: wasm-ld -shared --max-function-body-size=52 -o %t.52.wasm %t.o
# RUN: obj2yaml %t.52.wasm | FileCheck %s --check-prefix=ONE

## One byte less: four relocations in the first helper and one in the second.
# RUN: wasm-ld -shared --max-function-body-size=51 -o %t.51.wasm %t.o
# RUN: obj2yaml %t.51.wasm | FileCheck %s --check-prefix=SPLIT51

## Room for two relocations per function (1 + 2 * 10 + 1 = 22) but not three.
# RUN: wasm-ld -shared --max-function-body-size=22 -o %t.22.wasm %t.o
# RUN: obj2yaml %t.22.wasm | FileCheck %s --check-prefix=SPLIT22
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.22.wasm | FileCheck %s --check-prefix=DIS

## One byte less again: one relocation per function.
# RUN: wasm-ld -shared --max-function-body-size=21 -o %t.21.wasm %t.o
# RUN: obj2yaml %t.21.wasm | FileCheck %s --check-prefix=SPLIT21

## The helpers are local: --export-all must not export them (it does export
## hidden symbols), and a link that consumes a split shared library must not
## collide with them when it splits its own relocation code.
# RUN: wasm-ld -shared --export-all --max-function-body-size=22 -o %t.all.wasm %t.o
# RUN: obj2yaml %t.all.wasm | FileCheck %s --check-prefix=EXPORTALL
# RUN: wasm-ld -shared --max-function-body-size=22 -o %t.consumer.wasm %t.o %t.all.wasm

## The limit has to be positive.
# RUN: not wasm-ld -shared --max-function-body-size=0 -o %t.zero.wasm %t.o 2>&1 | FileCheck %s --check-prefix=ZERO
# RUN: not wasm-ld -shared --max-function-body-size=-1 -o %t.neg.wasm %t.o 2>&1 | FileCheck %s --check-prefix=ZERO

.section .data.ptrs,"",@
.globl ptrs
.p2align 2
ptrs:
  .int32 target
  .int32 target
  .int32 target
  .int32 target
  .int32 target
.size ptrs, 20

.section .data.target,"",@
.globl target
.p2align 2
target:
  .int32 42
.size target, 4

# ONE:          - Type:            EXPORT
# ONE:              Name:            __wasm_apply_data_relocs
# ONE-NEXT:         Kind:            FUNCTION
# ONE-NEXT:         Index:           1
# ONE-NOT:          __wasm_apply_data_relocs_
# ONE:          - Type:            CODE
# ONE-NEXT:       Functions:
# ONE-NEXT:         - Index:           0
# ONE-NEXT:           Locals:          []
# ONE-NEXT:           Body:            0B
# ONE-NEXT:         - Index:           1
# ONE-NEXT:           Locals:          []
# ONE-NEXT:           Body:            410023006A2302360200410423006A2302360200410823006A2302360200410C23006A2302360200411023006A23023602000B
# ONE-NEXT:     - Type:            DATA
# ONE-NOT:          __wasm_apply_data_relocs_

# SPLIT51:      - Type:            EXPORT
# SPLIT51:          Name:            __wasm_apply_data_relocs
# SPLIT51-NEXT:     Kind:            FUNCTION
# SPLIT51-NEXT:     Index:           1
# SPLIT51-NOT:      __wasm_apply_data_relocs_
# SPLIT51:      - Type:            CODE
# SPLIT51-NEXT:   Functions:
# SPLIT51-NEXT:     - Index:           0
# SPLIT51-NEXT:       Locals:          []
# SPLIT51-NEXT:       Body:            0B
# SPLIT51-NEXT:     - Index:           1
# SPLIT51-NEXT:       Locals:          []
# SPLIT51-NEXT:       Body:            100210030B
# SPLIT51-NEXT:     - Index:           2
# SPLIT51-NEXT:       Locals:          []
# SPLIT51-NEXT:       Body:            410023006A2302360200410423006A2302360200410823006A2302360200410C23006A23023602000B
# SPLIT51-NEXT:     - Index:           3
# SPLIT51-NEXT:       Locals:          []
# SPLIT51-NEXT:       Body:            411023006A23023602000B
# SPLIT51-NEXT: - Type:            DATA
# SPLIT51:          FunctionNames:
# SPLIT51:              Name:            __wasm_apply_data_relocs_0
# SPLIT51-NEXT:       - Index:           3
# SPLIT51-NEXT:         Name:            __wasm_apply_data_relocs_1
# SPLIT51-NEXT:     GlobalNames:

# SPLIT22:      - Type:            EXPORT
# SPLIT22:          Name:            __wasm_apply_data_relocs
# SPLIT22-NEXT:     Kind:            FUNCTION
# SPLIT22-NEXT:     Index:           1
# SPLIT22-NOT:      __wasm_apply_data_relocs_
# SPLIT22:      - Type:            CODE
# SPLIT22-NEXT:   Functions:
# SPLIT22-NEXT:     - Index:           0
# SPLIT22-NEXT:       Locals:          []
# SPLIT22-NEXT:       Body:            0B
# SPLIT22-NEXT:     - Index:           1
# SPLIT22-NEXT:       Locals:          []
# SPLIT22-NEXT:       Body:            1002100310040B
# SPLIT22-NEXT:     - Index:           2
# SPLIT22-NEXT:       Locals:          []
# SPLIT22-NEXT:       Body:            410023006A2302360200410423006A23023602000B
# SPLIT22-NEXT:     - Index:           3
# SPLIT22-NEXT:       Locals:          []
# SPLIT22-NEXT:       Body:            410823006A2302360200410C23006A23023602000B
# SPLIT22-NEXT:     - Index:           4
# SPLIT22-NEXT:       Locals:          []
# SPLIT22-NEXT:       Body:            411023006A23023602000B
# SPLIT22-NEXT: - Type:            DATA
# SPLIT22:          FunctionNames:
# SPLIT22-NEXT:       - Index:           0
# SPLIT22-NEXT:         Name:            __wasm_call_ctors
# SPLIT22-NEXT:       - Index:           1
# SPLIT22-NEXT:         Name:            __wasm_apply_data_relocs
# SPLIT22-NEXT:       - Index:           2
# SPLIT22-NEXT:         Name:            __wasm_apply_data_relocs_0
# SPLIT22-NEXT:       - Index:           3
# SPLIT22-NEXT:         Name:            __wasm_apply_data_relocs_1
# SPLIT22-NEXT:       - Index:           4
# SPLIT22-NEXT:         Name:            __wasm_apply_data_relocs_2
# SPLIT22-NEXT:     GlobalNames:

# DIS:      <__wasm_apply_data_relocs>:
# DIS-EMPTY:
# DIS-NEXT:   call 2
# DIS-NEXT:   call 3
# DIS-NEXT:   call 4
# DIS-NEXT:   end
# DIS:      <__wasm_apply_data_relocs_0>:
# DIS-EMPTY:
# DIS-NEXT:   i32.const 0
# DIS-NEXT:   global.get 0
# DIS-NEXT:   i32.add
# DIS-NEXT:   global.get 2
# DIS-NEXT:   i32.store 0
# DIS-NEXT:   i32.const 4
# DIS-NEXT:   global.get 0
# DIS-NEXT:   i32.add
# DIS-NEXT:   global.get 2
# DIS-NEXT:   i32.store 0
# DIS-NEXT:   end
# DIS:      <__wasm_apply_data_relocs_1>:
# DIS-EMPTY:
# DIS-NEXT:   i32.const 8
# DIS:        i32.const 12
# DIS:        end
# DIS:      <__wasm_apply_data_relocs_2>:
# DIS-EMPTY:
# DIS-NEXT:   i32.const 16
# DIS-NEXT:   global.get 0
# DIS-NEXT:   i32.add
# DIS-NEXT:   global.get 2
# DIS-NEXT:   i32.store 0
# DIS-NEXT:   end

# SPLIT21:      - Type:            CODE
# SPLIT21-NEXT:   Functions:
# SPLIT21-NEXT:     - Index:           0
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            0B
# SPLIT21-NEXT:     - Index:           1
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            100210031004100510060B
# SPLIT21-NEXT:     - Index:           2
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            410023006A23023602000B
# SPLIT21-NEXT:     - Index:           3
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            410423006A23023602000B
# SPLIT21-NEXT:     - Index:           4
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            410823006A23023602000B
# SPLIT21-NEXT:     - Index:           5
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            410C23006A23023602000B
# SPLIT21-NEXT:     - Index:           6
# SPLIT21-NEXT:       Locals:          []
# SPLIT21-NEXT:       Body:            411023006A23023602000B
# SPLIT21-NEXT: - Type:            DATA
# SPLIT21:          Name:            __wasm_apply_data_relocs_4
# SPLIT21-NEXT:     GlobalNames:

# EXPORTALL:      - Type:            EXPORT
# EXPORTALL:          Name:            __wasm_apply_data_relocs
# EXPORTALL-NOT:      __wasm_apply_data_relocs_
# EXPORTALL:      - Type:            CODE
# EXPORTALL:          FunctionNames:
# EXPORTALL:              Name:            __wasm_apply_data_relocs_0
# EXPORTALL:              Name:            __wasm_apply_data_relocs_1
# EXPORTALL:              Name:            __wasm_apply_data_relocs_2

# ZERO: error: --max-function-body-size=N must be greater than 0
