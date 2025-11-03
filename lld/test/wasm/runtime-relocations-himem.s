## Verifies runtime relocation code for addresses over 2gb works correctly.
## We have had issues with LEB encoding of address over 2gb in i32.const
## instruction leading to invalid binaries.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --global-base=2147483648 --experimental-pic --unresolved-symbols=import-dynamic -no-gc-sections --shared-memory --no-entry -o %t.wasm %t.o
# XUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --

.globl tls_sym
.globl data_sym
.globl _start
.globaltype __tls_base, i32

_start:
  .functype _start () -> ()
  global.get __tls_base
  i32.const tls_sym@TLSREL
  i32.add
  drop
  i32.const data_sym
  drop
  end_function

.section tls_sec,"T",@
.p2align  2
tls_sym:
  .int32 0
  .int32 extern_sym
  .size tls_sym, 8

.section data_sec,"",@
.p2align  2
data_sym:
  .int32 0
  .int32 extern_sym
  .size data_sym, 8

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# CHECK: <__wasm_apply_data_relocs>:
# CHECK-EMPTY:
# CHECK-NEXT:  i32.const -2147483636
# CHECK-NEXT:  global.get 0
# CHECK-NEXT:  i32.store 0
# CHECK-NEXT:  end

# CHECK: <__wasm_apply_tls_relocs>:
# CHECK-EMPTY:
# CHECK-NEXT:  i32.const -2147483644
# CHECK-NEXT:  global.get 0
# CHECK-NEXT:  i32.store 0
# CHECK-NEXT:  end
