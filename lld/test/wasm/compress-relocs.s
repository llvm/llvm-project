# RUN: llvm-mc -mattr=+reference-types,+exception-handling -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --export-dynamic -o %t.wasm %t.o
# RUN: llvm-objdump -d %t.wasm | FileCheck %s
# RUN: wasm-ld --export-dynamic -O2 -o %t-opt.wasm %t.o
# RUN: llvm-objdump -d %t-opt.wasm | FileCheck %s
# RUN: not wasm-ld --compress-relocations -o %t-compressed.wasm %t.o 2>&1 | FileCheck %s -check-prefix=ERROR
# RUN: wasm-ld --export-dynamic --strip-debug --compress-relocations -o %t-compressed.wasm %t.o
# RUN: llvm-objdump -d %t-compressed.wasm | FileCheck %s -check-prefix=COMPRESS

  .globl _start
_start:
  .functype _start () -> ()
  end_function

  .globl func_ret_i64
func_ret_i64:
  .functype func_ret_i64 () -> (i64)
  i64.const 1
  end_function

  .globl func_ret_i32
func_ret_i32:
  .functype func_ret_i32 () -> (i32)
  i32.const 2
  end_function

i32_global:
  .globaltype i32_global, i32

i32_tag:
  .tagtype i32_tag i32

  .globl test_memory_and_indirect_call_relocs
test_memory_and_indirect_call_relocs:
  .functype test_memory_and_indirect_call_relocs () -> ()
  i32.const indirect_func_ret_i64 # R_WASM_MEMORY_ADDR_SLEB
  i32.load  0
  call_indirect () -> (i64)       # R_WASM_TYPE_INDEX_LEB, R_WASM_TABLE_NUMBER_LEB
  drop
  i32.const 0
  i32.load  indirect_func_ret_i32 # R_WASM_MEMORY_ADDR_LEB
  call_indirect () -> (i32)
  drop
  i32.const func_ret_i64          # R_WASM_TABLE_INDEX_SLEB
  call_indirect () -> (i64)
  drop
  end_function

# CHECK:    test_memory_and_indirect_call_relocs
# CHECK:      41 90 88 80 80 00                 i32.const      1040
# CHECK:      11 80 80 80 80 00 80 80 80 80 00  call_indirect  0
# CHECK:      28 02 94 88 80 80 00              i32.load       1044
# CHECK:      11 81 80 80 80 00 80 80 80 80 00  call_indirect  1
# CHECK:      41 81 80 80 80 00                 i32.const      1
# CHECK:      11 80 80 80 80 00 80 80 80 80 00  call_indirect  0
# COMPRESS: test_memory_and_indirect_call_relocs
# COMPRESS:   41 90 08                          i32.const      1040
# COMPRESS:   11 00 00                          call_indirect  0
# COMPRESS:   28 02 94 08                       i32.load       1044
# COMPRESS:   11 01 00                          call_indirect  1
# COMPRESS:   41 01                             i32.const      1
# COMPRESS:   11 00 00                          call_indirect  0

  .globl test_simple_index_relocs
test_simple_index_relocs:
  .functype test_simple_index_relocs () -> ()
  call       func_ret_i32 # R_WASM_FUNCTION_INDEX_LEB
  global.set i32_global   # R_WASM_GLOBAL_INDEX_LEB
  i32.const  0
  throw      i32_tag      # R_WASM_TAG_INDEX_LEB
  end_function

# CHECK:    test_simple_index_relocs
# CHECK:      10 82 80 80 80 00  call        2
# CHECK:      24 81 80 80 80 00  global.set  1
# CHECK:      08 80 80 80 80 00  throw       0
# COMPRESS: test_simple_index_relocs
# COMPRESS:   10 02              call        2
# COMPRESS:   24 01              global.set  1
# COMPRESS:   08 00              throw       0

  .globl test_relative_relocs
test_relative_relocs:
  .functype test_relative_relocs () -> ()
  i32.const indirect_func_ret_i64@MBREL # R_WASM_MEMORY_ADDR_REL_SLEB
  drop
  i32.const func_ret_i32@TBREL          # R_WASM_TABLE_INDEX_REL_SLEB
  drop
  i32.const i32_tls_data@TLSREL         # R_WASM_MEMORY_ADDR_TLS_SLEB
  drop
  end_function

# CHECK:    test_relative_relocs
# CHECK:      41 90 88 80 80 00  i32.const  1040
# CHECK:      41 81 80 80 80 00  i32.const  1
# CHECK:      41 83 80 80 80 00  i32.const  3
# COMPRESS: test_relative_relocs
# COMPRESS:   41 90 08           i32.const  1040
# COMPRESS:   41 01              i32.const  1
# COMPRESS:   41 03              i32.const  3

# ERROR: wasm-ld: error: --compress-relocations is incompatible with output debug information. Please pass --strip-debug or --strip-all

  .section .tdata,"T",@
  .int8 0
  .int8 0
  .int8 0
i32_tls_data:
  .int32 65
  .size  i32_tls_data, 4

  .section .data,"",@
  .p2align 4
indirect_func_ret_i64:
  .int32 func_ret_i64
  .size  indirect_func_ret_i64, 4

indirect_func_ret_i32:
  .int32 func_ret_i32
  .size  indirect_func_ret_i32, 4

.section .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
