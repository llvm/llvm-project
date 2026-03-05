# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -no-gc-sections -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-print-imm-hex --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=DIS

.functype       __wasm_component_model_builtin_context_get_1 () -> (i32)
.import_module  __wasm_component_model_builtin_context_get_1, "$root"
.import_name    __wasm_component_model_builtin_context_get_1, "[context-get-1]"

.globl _start
_start:
  .functype _start () -> (i32)
  call __wasm_component_model_builtin_context_get_1
  i32.const tls1@TLSREL
  i32.add
  i32.load 0
  call __wasm_component_model_builtin_context_get_1
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
  .int8 30
  .ascii  "component-model-thread-context"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"


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
# DIS-EMPTY:
# DIS-NEXT:       i32.const       65536
# DIS-NEXT:       i32.const       65536
# DIS-NEXT:       call    1
# DIS-NEXT:       i32.const       0
# DIS-NEXT:       i32.const       8
# DIS-NEXT:       memory.init     0, 0
# DIS-NEXT:       end

# DIS-LABEL: <_start>:
# DIS-EMPTY:
# DIS-NEXT:       call    0
# DIS-NEXT:       i32.const       0
# DIS-NEXT:       i32.add 
# DIS-NEXT:       i32.load        0
# DIS-NEXT:       call    0
# DIS-NEXT:       i32.const       4
# DIS-NEXT:       i32.add 
# DIS-NEXT:       i32.load        0
# DIS-NEXT:       i32.add 
# DIS-NEXT:       end
