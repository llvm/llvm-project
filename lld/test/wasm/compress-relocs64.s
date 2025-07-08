# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown %s -o %t.o
# RUN: wasm-ld -mwasm64 --export-dynamic -o %t.wasm %t.o
# RUN: llvm-objdump -d %t.wasm | FileCheck %s
# RUN: wasm-ld -mwasm64 --export-dynamic -O2 -o %t-opt.wasm %t.o
# RUN: llvm-objdump -d %t-opt.wasm | FileCheck %s
# RUN: wasm-ld -mwasm64 --export-dynamic --strip-debug --compress-relocations -o %t-compressed.wasm %t.o
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

  .globl test_memory_and_indirect_call_relocs
test_memory_and_indirect_call_relocs:
  .functype test_memory_and_indirect_call_relocs () -> ()
  i64.const indirect_func_ret_i64 # R_WASM_MEMORY_ADDR_SLEB64
  drop
  i64.const 0
  i64.load  indirect_func_ret_i32 # R_WASM_MEMORY_ADDR_LEB64
  drop
  i64.const func_ret_i64          # R_WASM_TABLE_INDEX_SLEB64
  drop
  end_function

# CHECK:    test_memory_and_indirect_call_relocs
# CHECK:      42 90 88 80 80 80 80 80 80 80 00     i64.const 1040
# CHECK:      29 03 98 88 80 80 80 80 80 80 80 00  i64.load  1048
# CHECK:      42 81 80 80 80 80 80 80 80 80 00     i64.const 1
# COMPRESS: test_memory_and_indirect_call_relocs
# COMPRESS:   42 90 08                             i64.const 1040
# COMPRESS:   29 03 98 08                          i64.load  1048
# COMPRESS:   42 01                                i64.const 1

  .globl test_relative_relocs
test_relative_relocs:
  .functype test_relative_relocs () -> ()
  i64.const indirect_func_ret_i64@MBREL # R_WASM_MEMORY_ADDR_REL_SLEB64
  drop
  i64.const func_ret_i32@TBREL          # R_WASM_TABLE_INDEX_REL_SLEB64
  drop
  i64.const i32_tls_data@TLSREL         # R_WASM_MEMORY_ADDR_TLS_SLEB64
  drop
  end_function

# CHECK:    test_relative_relocs
# CHECK:      42 90 88 80 80 80 80 80 80 80 00  i64.const 1040
# CHECK:      42 81 80 80 80 80 80 80 80 80 00  i64.const 1
# CHECK:      42 83 80 80 80 80 80 80 80 80 00  i64.const 3
# COMPRESS: test_relative_relocs
# COMPRESS:   42 90 08                          i64.const 1040
# COMPRESS:   42 01                             i64.const 1
# COMPRESS:   42 03                             i64.const 3

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
  .int64 func_ret_i64
  .size  indirect_func_ret_i64, 8

indirect_func_ret_i32:
  .int64 func_ret_i32
  .size  indirect_func_ret_i32, 8

.section .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
