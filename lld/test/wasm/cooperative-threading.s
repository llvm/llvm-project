# Test that --cooperative-threading uses the libcall ABI naming for
# thread-context globals (__init_stack_pointer, __init_tls_base, etc.) and
# works without --shared-memory and atomics.

# RUN: llvm-mc -mattr=+call-indirect-overlong -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --cooperative-threading -no-gc-sections -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-print-imm-hex --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=DIS

# Test that --cooperative-threading and --shared-memory are mutually exclusive.
# RUN: not wasm-ld --cooperative-threading --shared-memory %t.o -o %t2.wasm 2>&1 | FileCheck %s --check-prefix=INCOMPAT
# INCOMPAT: --cooperative-threading is incompatible with --shared-memory

.globl __indirect_function_table
.tabletype __indirect_function_table, funcref

.globl         __wasm_get_tls_base
__wasm_get_tls_base:
  .functype   __wasm_get_tls_base () -> (i32)
  i32.const 0
  end_function

.globl do_call_indirect
do_call_indirect:
  .functype do_call_indirect () -> ()
  i32.const 1
  call_indirect __indirect_function_table, () -> ()
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

.section  .data.bar,"",@
.globl  bar
.p2align  2
bar:
  .int32  42
  .size bar, 4

.section  .bss.foo,"",@
.globl  foo
.p2align  2
foo:
  .int32  0
  .size foo, 4

.section  .rodata.baz,"",@
.globl  baz
.p2align  2
baz:
  .int32  1
  .size baz, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 7
  .ascii  "atomics"

# CHECK:      - Type:            TABLE
# CHECK-NEXT:   Tables:
# CHECK-NEXT:     - Index:           0
# CHECK-NEXT:       ElemType:        FUNCREF

# Memory must NOT be marked as shared.
# CHECK:      - Type:            MEMORY
# CHECK-NEXT:   Memories:
# CHECK-NEXT:     - Minimum:         0x2
# CHECK-NOT:       Shared

# Ensure __init_stack_pointer, __init_tls_base, and __tls_size are all correct.
# CHECK:      - Type:            GLOBAL
# CHECK-NEXT:   Globals:
# CHECK-NEXT:     - Index:           0
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           65536
# CHECK-NEXT:     - Index:           1
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         true
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           65544
# CHECK-NEXT:     - Index:           2
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           8

# The function table is exported by default.
# CHECK:      - Type:            EXPORT
# CHECK:          - Name:            __indirect_function_table
# CHECK-NEXT:       Kind:            TABLE
# CHECK-NEXT:       Index:           0

# Only TLS needs a passive data segment; .rodata and .data stay active and
# .bss gets no segment at all since memory is only instantiated once and
# starts zeroed. The TLS segment is sorted last.
# CHECK:        - Type:            DATACOUNT
# CHECK-NEXT:     Count:           3

# CHECK:        - Type:            DATA{{$}}
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   8
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:         Content:         '01000000'
# CHECK-NEXT:       - SectionOffset:   19
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65540
# CHECK-NEXT:         Content:         2A000000
# CHECK-NEXT:       - SectionOffset:   25
# CHECK-NEXT:         InitFlags:       1
# CHECK-NEXT:         Content:         '0100000002000000'
# CHECK-NEXT:   - Type:            CUSTOM

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
# DIS-EMPTY:
# DIS-NEXT:      i32.const       65544
# DIS-NEXT:      i32.const       65544
# DIS-NEXT:      call    0
# DIS-NEXT:      i32.const       0
# DIS-NEXT:      i32.const       8
# DIS-NEXT:      memory.init     2, 0
# DIS-NEXT:      end

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

# When the table is imported instead there is no need to also export it.
# RUN: wasm-ld --cooperative-threading --import-table -no-gc-sections -o %t3.wasm %t.o
# RUN: obj2yaml %t3.wasm | FileCheck %s --check-prefix=IMPORT-TABLE

# IMPORT-TABLE:      - Type:            IMPORT
# IMPORT-TABLE:          - Module:          env
# IMPORT-TABLE-NEXT:       Field:           __indirect_function_table
# IMPORT-TABLE-NEXT:       Kind:            TABLE
# IMPORT-TABLE-NOT:        Kind:            TABLE

# Test --cooperative-threading combined with PIC output.
# RUN: wasm-ld -shared --cooperative-threading -no-gc-sections -o %t.so %t.o
# RUN: obj2yaml %t.so | FileCheck %s --check-prefix=PIC
# RUN: llvm-objdump --disassemble-symbols=__wasm_init_memory --no-show-raw-insn --no-leading-addr %t.so | FileCheck %s --check-prefix=PIC-DIS

# The stack pointer is imported under the libcall ABI name and
# __wasm_set_tls_base is imported for TLS initialization.
# PIC:       - Type:            IMPORT
# PIC:           Field:           __init_stack_pointer
# PIC-NEXT:      Kind:            GLOBAL
# PIC-NEXT:      GlobalType:      I32
# PIC-NEXT:      GlobalMutable:   false
# PIC:           Field:           __memory_base
# PIC:           Field:           __table_base
# PIC:           Field:           __wasm_set_tls_base
# PIC-NEXT:      Kind:            FUNCTION

# The PIC `__init_tls_base` global (global 3) is mutable and initialized to
# 0 since its final value is calculated once `__memory_base` is provided.
# PIC:       - Type:            GLOBAL
# PIC-NEXT:    Globals:
# PIC-NEXT:      - Index:           3
# PIC-NEXT:        Type:            I32
# PIC-NEXT:        Mutable:         true
# PIC-NEXT:        InitExpr:
# PIC-NEXT:          Opcode:          I32_CONST
# PIC-NEXT:          Value:           0
# PIC-NEXT:      - Index:           4
# PIC-NEXT:        Type:            I32
# PIC-NEXT:        Mutable:         false
# PIC-NEXT:        InitExpr:
# PIC-NEXT:          Opcode:          I32_CONST
# PIC-NEXT:          Value:           8

# In PIC mode the active .rodata and .data segments are combined into a single
# active segment at __memory_base; the TLS segment remains passive.
# PIC:       - Type:            DATACOUNT
# PIC-NEXT:    Count:           2
# PIC:       - Type:            DATA{{$}}
# PIC-NEXT:    Segments:
# PIC-NEXT:      - SectionOffset:   6
# PIC-NEXT:        InitFlags:       0
# PIC-NEXT:        Offset:
# PIC-NEXT:          Opcode:          GLOBAL_GET
# PIC-NEXT:          Index:           {{[0-9]+}}
# PIC-NEXT:        Content:         010000002A000000
# PIC-NEXT:      - SectionOffset:   {{[0-9]+}}
# PIC-NEXT:        InitFlags:       1
# PIC-NEXT:        Content:         '0100000002000000'
# PIC-NEXT:  - Type:            CUSTOM

# PIC:       GlobalNames:
# PIC-NEXT:      - Index:           0
# PIC-NEXT:        Name:            __init_stack_pointer
# PIC-NEXT:      - Index:           1
# PIC-NEXT:        Name:            __memory_base
# PIC-NEXT:      - Index:           2
# PIC-NEXT:        Name:            __table_base
# PIC-NEXT:      - Index:           3
# PIC-NEXT:        Name:            __init_tls_base
# PIC-NEXT:      - Index:           4
# PIC-NEXT:        Name:            __tls_size
# PIC-NEXT:      - Index:           5
# PIC-NEXT:        Name:            __tls_align

# Memory initialization in PIC mode has a few responsibilities: it calculates
# the TLS address and puts it in a local, stores it into the __init_tls_base
# global, `__wasm_set_tls_base` is called, TLS is initialized, and then finally
# BSS is zero'd out.
# PIC-DIS:      <__wasm_init_memory>:
# PIC-DIS-NEXT:   .local i32
# PIC-DIS-NEXT:   i32.const 8
# PIC-DIS-NEXT:   global.get 1
# PIC-DIS-NEXT:   i32.add
# PIC-DIS-NEXT:   local.tee 0
# PIC-DIS-NEXT:   local.get 0
# PIC-DIS-NEXT:   global.set 3
# PIC-DIS-NEXT:   call {{[0-9]+}}
# PIC-DIS-NEXT:   local.get 0
# PIC-DIS-NEXT:   i32.const 0
# PIC-DIS-NEXT:   i32.const 8
# PIC-DIS-NEXT:   memory.init 1, 0
# PIC-DIS-NEXT:   i32.const 16
# PIC-DIS-NEXT:   global.get {{[0-9]+}}
# PIC-DIS-NEXT:   i32.add
# PIC-DIS-NEXT:   i32.const 0
# PIC-DIS-NEXT:   i32.const 4
# PIC-DIS-NEXT:   memory.fill 0
# PIC-DIS-NEXT:   end
