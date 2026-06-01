# Test that --cooperative-multithreading uses the libcall ABI naming for
# thread-context globals (__init_stack_pointer, __init_tls_base, etc.) and
# works without --shared-memory and atomics.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --cooperative-multithreading -no-gc-sections -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-print-imm-hex --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=DIS

# Test that --cooperative-multithreading and --shared-memory are mutually exclusive.
# RUN: not wasm-ld --cooperative-multithreading --shared-memory %t.o -o %t2.wasm 2>&1 | FileCheck %s --check-prefix=INCOMPAT
# INCOMPAT: --cooperative-multithreading is incompatible with --shared-memory

.globl         __wasm_get_tls_base
__wasm_get_tls_base:
    .functype   __wasm_get_tls_base () -> (i32)
    i32.const 0
    end_function

.globl _start
_start:
  .functype _start () -> (i32)
  call __wasm_get_tls_base
  i32.const tls1@TLSREL
  i32.add
  i32.load 0
  call __wasm_get_tls_base
  i32.const tls2@TLSREL
  i32.add
  i32.load 0
  i32.add
  end_function

.section  .tdata.tls1,"",@
.globl  tls1
tls1:
  .int32  1
  .size tls1, 4

.section  .tdata.tls2,"",@
.globl  tls2
tls2:
  .int32  2
  .size tls2, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 7
  .ascii  "atomics"

# Memory must NOT be marked as shared.
# CHECK:      - Type:            MEMORY
# CHECK-NEXT:   Memories:
# CHECK-NEXT:     - Minimum:         0x2
# CHECK-NOT:       Shared:          false

# Globals should use the libcall ABI naming, not the global ABI.
# CHECK:      GlobalNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __init_stack_pointer
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            __init_tls_base
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            __tls_size
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            __tls_align

# DIS-LABEL: <__wasm_init_memory>:

# DIS-LABEL: <_start>:
# DIS-EMPTY:
# DIS-NEXT:       call    {{[0-9]+}}
# DIS-NEXT:       i32.const       0
# DIS-NEXT:       i32.add
# DIS-NEXT:       i32.load        0
# DIS-NEXT:       call    {{[0-9]+}}
# DIS-NEXT:       i32.const       4
# DIS-NEXT:       i32.add
# DIS-NEXT:       i32.load        0
# DIS-NEXT:       i32.add
# DIS-NEXT:       end
