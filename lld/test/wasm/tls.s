# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globaltype __tls_base, i32
.globaltype __tls_align, i32, immutable

.globl tls1_addr
tls1_addr:
  .functype tls1_addr () -> (i32)
  global.get __tls_base
  i32.const tls1@TLSREL
  i32.add
  end_function

.globl tls2_addr
tls2_addr:
  .functype tls2_addr () -> (i32)
  global.get __tls_base
  i32.const tls2@TLSREL
  i32.add
  end_function

.globl tls3_addr
tls3_addr:
  .functype tls3_addr () -> (i32)
  global.get __tls_base
  i32.const tls3
  i32.add
  end_function

.globl tls_align
tls_align:
  .functype tls_align () -> (i32)
  global.get __tls_align
  end_function

# TLS symbols can also be accessed by `global.get tls1@GOT@TLS`
# which is the pattern emitted for non-DSO-local symbols.
# In this case the global that holds that address must be
# initialized by `__wasm_apply_global_tls_relocs` which is
# called by `__wasm_init_tls`.
.globl tls1_got_addr
tls1_got_addr:
  .functype tls1_got_addr () -> (i32)
  global.get tls1@GOT@TLS
  end_function

.section  .bss.no_tls,"",@
.globl  no_tls
.p2align  2
no_tls:
  .int32  0
  .size no_tls, 4

// Older versions of LLVM did not use the "T" flag so we need to support
// infering TLS from the name alone.
.section  .tdata.tls1,"",@
.globl  tls1
.p2align  2
tls1:
  .int32  1
  .size tls1, 4

.section  sec_tls2,"T",@
.globl  tls2
.p2align  2
tls2:
  .int32  1
  .size tls2, 4

.section  sec_tls3,"T",@
.globl  tls3
.p2align  2
tls3:
  .int32  0
  .size tls3, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck --check-prefix=ASM %s --

# RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --no-merge-data-segments --no-entry -o %t2.wasm %t.o
# RUN: obj2yaml %t2.wasm | FileCheck %s

# CHECK:      - Type:            GLOBAL
# __stack_pointer
# CHECK-NEXT:   Globals:
# CHECK-NEXT:     - Index:           0
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         true
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           66592

# __tls_base
# CHECK-NEXT:     - Index:           1
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         true
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           0

# __tls_size
# CHECK-NEXT:     - Index:           2
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           12

# __tls_align
# CHECK-NEXT:     - Index:           3
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           4


# ASM-LABEL: <__wasm_init_tls>:
# ASM-EMPTY:
# ASM-NEXT:   local.get 0
# ASM-NEXT:   global.set 1
# ASM-NEXT:   local.get 0
# ASM-NEXT:   i32.const 0
# ASM-NEXT:   i32.const 12
# ASM-NEXT:   memory.init 0, 0
# call to __wasm_apply_global_tls_relocs
# ASM-NEXT:   call 3
# ASM-NEXT:   end

# ASM-LABEL: <__wasm_apply_global_tls_relocs>:
# ASM-EMPTY:
# ASM-NEXT:   global.get      1
# ASM-NEXT:   i32.const       0
# ASM-NEXT:   i32.add
# ASM-NEXT:   global.set      4
# ASM-NEXT:   end

# ASM-LABEL: <tls1_addr>:
# ASM-EMPTY:
# ASM-NEXT:   global.get 1
# ASM-NEXT:   i32.const 0
# ASM-NEXT:   i32.add
# ASM-NEXT:   end

# ASM-LABEL: <tls2_addr>:
# ASM-EMPTY:
# ASM-NEXT:   global.get 1
# ASM-NEXT:   i32.const 4
# ASM-NEXT:   i32.add
# ASM-NEXT:   end

# ASM-LABEL: <tls3_addr>:
# ASM-EMPTY:
# ASM-NEXT:   global.get 1
# ASM-NEXT:   i32.const 8
# ASM-NEXT:   i32.add
# ASM-NEXT:   end

# ASM-LABEL: <tls_align>:
# ASM-EMPTY:
# ASM-NEXT:   global.get 3
# ASM-NEXT:   end

# Also verify TLS usage with --relocatable
# RUN: wasm-ld --relocatable -o %t3.wasm %t.o
# RUN: obj2yaml %t3.wasm | FileCheck %s --check-prefix=RELOC

# RELOC:       - Type:            IMPORT
# RELOC-NEXT:    Imports:
# RELOC-NEXT:      - Module:          env
# RELOC-NEXT:        Field:           __tls_base
# RELOC-NEXT:        Kind:            GLOBAL
# RELOC-NEXT:        GlobalType:      I32
# RELOC-NEXT:        GlobalMutable:   true
# RELOC-NEXT:      - Module:          env
# RELOC-NEXT:        Field:           __tls_align
# RELOC-NEXT:        Kind:            GLOBAL
# RELOC-NEXT:        GlobalType:      I32
# RELOC-NEXT:        GlobalMutable:   false

# RELOC:         GlobalNames:
# RELOC-NEXT:      - Index:           0
# RELOC-NEXT:        Name:            __tls_base
# RELOC-NEXT:      - Index:           1
# RELOC-NEXT:        Name:            __tls_align
# RELOC-NEXT:      - Index:           2
# RELOC-NEXT:        Name:            GOT.data.internal.tls1
# RELOC-NEXT:    DataSegmentNames:
# RELOC-NEXT:      - Index:           0
# RELOC-NEXT:        Name:            .tdata
# RELOC-NEXT:      - Index:           1
# RELOC-NEXT:        Name:            .bss.no_tls
