# Regression test based on https://github.com/llvm/llvm-project/issues/54386
# Test that that linker synthetic functions such as __wasm_tls_init and
# __wasm_apply_global_tls_relocs can be created successfully in programs
# that don't reference __tls_base or __wasm_tls_init.  These function both
# reference __tls_base which need to be marks as alive if they are generated.

# This is very basic TLS-using program that doesn't reference any of the
# linker-generated symbols.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -no-gc-sections --shared-memory -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

.globl _start
_start:
  .functype _start () -> (i32)
  global.get tls_sym@GOT@TLS
  end_function

.section  .tdata.tls_sym,"",@
.globl  tls_sym
.p2align  2
tls_sym:
  .int32  1
  .size tls_sym, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __wasm_call_ctors
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            __wasm_init_tls
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            __wasm_init_memory
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            __wasm_apply_global_tls_relocs
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        Name:            _start

# DIS:       <__wasm_init_tls>:
# DIS:        local.get 0
# DIS-NEXT:   global.set  1
# DIS-NEXT:   local.get 0
# DIS-NEXT:   i32.const 0
# DIS-NEXT:   i32.const 4
# DIS-NEXT:   memory.init 0, 0
# DIS-NEXT:   call  3
# DIS-NEXT:   end

# DIS:      <__wasm_apply_global_tls_relocs>:
# DIS:        global.get  1
# DIS-NEXT:   i32.const 0
# DIS-NEXT:   i32.add
# DIS-NEXT:   global.set  4
# DIS-NEXT:   end

# DIS:      <_start>:
# DIS:        global.get  4
# DIS-NEXT:   end

