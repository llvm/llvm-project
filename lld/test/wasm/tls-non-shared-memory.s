# Test that linking without shared memory causes __tls_base to be
# internalized.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globaltype __tls_base, i32

.globl get_tls1
get_tls1:
  .functype get_tls1 () -> (i32)
  global.get __tls_base
  i32.const tls1@TLSREL
  i32.add
  end_function

.globl get_tls1_got
get_tls1_got:
  .functype get_tls1_got () -> (i32)
  global.get tls1@GOT@TLS
  end_function

.section  .data.no_tls,"",@
.globl  no_tls
.p2align  2
no_tls:
  .int32  42
  .size no_tls, 4

.section  .tdata.tls1,"T",@
.globl  tls1
.p2align  2
tls1:
  .int32  43
  .size tls1, 2

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld --no-gc-sections --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=get_tls1 --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

# RUN: wasm-ld --experimental-pic -shared -o %t.so %t.o
# RUN: obj2yaml %t.so | FileCheck %s --check-prefixes=SHARED,PIC

# RUN: wasm-ld --experimental-pic --no-gc-sections --no-entry -pie -o %t-pie.wasm %t.o
# RUN: obj2yaml %t-pie.wasm | FileCheck %s --check-prefixes=PIE,PIC

# RUN: wasm-ld --experimental-pic --features=atomics,bulk-memory,extended-const --no-gc-sections --no-entry -pie -o %t-extended-const.wasm %t.o
# RUN: obj2yaml %t-extended-const.wasm | FileCheck %s --check-prefixes=EXT-CONST

#      CHECK:   - Type:            GLOBAL
# __stack_pointer
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66576
# __tls_base
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1024
# GOT.data.internal.tls1
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1024
# CHECK-NEXT:   - Type:            EXPORT

#      CHECK:  - Type:            DATA
# .data
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - SectionOffset:   7
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1024
# CHECK-NEXT:        Content:         2B000000
# .tdata
# CHECK-NEXT:      - SectionOffset:   17
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1028
# CHECK-NEXT:        Content:         2A000000
# CHECK-NEXT:  - Type:            CUSTOM

# The constant value here which we add to `__tls_base` should not be absolute
# but relative to `__tls_base`, in this case zero rather than 1024.
# DIS:      <get_tls1>:
# DIS-EMPTY:
# DIS-NEXT:  global.get 1
# DIS-NEXT:  i32.const 0
# DIS-NEXT:  i32.add
# DIS-NEXT:  end

# In PIC mode we expect TLS data and non-TLS data to be merged into
# a single segment which is initialized via the  __memory_base import

#      SHARED:  - Type:            IMPORT
# SHARED-NEXT:    Imports:
# SHARED-NEXT:      - Module:          env
# SHARED-NEXT:        Field:           memory
# SHARED-NEXT:        Kind:            MEMORY
# SHARED-NEXT:        Memory:
# SHARED-NEXT:          Minimum:         0x1
# SHARED-NEXT:      - Module:          env
# SHARED-NEXT:        Field:           __memory_base
# SHARED-NEXT:        Kind:            GLOBAL
# SHARED-NEXT:        GlobalType:      I32

# In SHARED mode we export the address of all data symbols.
#      SHARED:   - Type:            EXPORT
# SHARED-NEXT:     Exports:
#      SHARED:     - Name:            tls1
# SHARED-NEXT:       Kind:            GLOBAL
#      SHARED:     - Name:            no_tls
# SHARED-NEXT:       Kind:            GLOBAL

# In PIE mode we don't export data address by default.
#      PIE:   - Type:            EXPORT
# PIE-NEXT:     Exports:
# PIE-NEXT:       - Name:            memory
# PIE-NEXT:         Kind:            MEMORY
# PIE-NEXT:         Index:           0
# PIE-NEXT:   - Type:

# .tdata and .data are combined into single segment in PIC mode.
#      PIC:  - Type:            DATA
# PIC-NEXT:    Segments:
# PIC-NEXT:      - SectionOffset:   6
# PIC-NEXT:        InitFlags:       0
# PIC-NEXT:        Offset:
# PIC-NEXT:          Opcode:          GLOBAL_GET
# PIC-NEXT:          Index:           {{\d*}}
# PIC-NEXT:        Content:         2B0000002A000000
# PIC-NEXT:  - Type:            CUSTOM

# Unless we have extended-const, in which case the merging is not needed.
# The first segment is placed directly at `__memory_base` and the second
# one is offset from `__memory_base` using `i32.add` and a constant.

#      EXT-CONST:  - Type:            DATA
# EXT-CONST-NEXT:    Segments:
# EXT-CONST-NEXT:      - SectionOffset:   6
# EXT-CONST-NEXT:        InitFlags:       0
# EXT-CONST-NEXT:        Offset:
# EXT-CONST-NEXT:          Opcode:          GLOBAL_GET
# EXT-CONST-NEXT:          Index:           1
# EXT-CONST-NEXT:        Content:         2B000000
# EXT-CONST-NEXT:      - SectionOffset:   18
# EXT-CONST-NEXT:        InitFlags:       0
# EXT-CONST-NEXT:        Offset:
# EXT-CONST-NEXT:          Extended:        true
# This instruction sequence decodes to:
# (global.get[0x23] 0x1 i32.const[0x41] 0x04 i32.add[0x6A] end[0x0b])
# EXT-CONST-NEXT:          Body:            230141046A0B
# EXT-CONST-NEXT:        Content:         2A000000
