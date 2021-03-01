# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hello.s -o %t.hello32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t32.o
# RUN: wasm-ld -m wasm32 -no-gc-sections --export=__data_end --export=__heap_base --allow-undefined --no-entry -o %t32.wasm %t32.o %t.hello32.o
# RUN: obj2yaml %t32.wasm | FileCheck --check-prefixes CHECK,CHK32 %s
#
# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown %p/Inputs/hello.s -o %t.hello64.o
# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown %s -o %t64.o
# RUN: wasm-ld -m wasm64 -no-gc-sections --export=__data_end --export=__heap_base --allow-undefined --no-entry -o %t64.wasm %t64.o %t.hello64.o
# RUN: obj2yaml %t64.wasm | FileCheck --check-prefixes CHECK,CHK64 %s

        .section .data.foo,"",@
        .globl  foo
        .hidden  foo
        .p2align        2
foo:
        .int32  1
        .size   foo, 4


        .section .data.aligned_bar,"",@
        .globl  aligned_bar
        .hidden  aligned_bar
        .p2align        4
aligned_bar:
        .int32  3
        .size   aligned_bar, 4


        .section .data.external_ref,"",@
        .globl  external_ref
        .p2align        3
external_ref:
        .int32  hello_str
        .size   external_ref, 4


        .section .bss.local_struct,"",@
        .globl  local_struct
        .hidden  local_struct
        .p2align        2
local_struct:
        .skip   8
        .size   local_struct, 8


        .section .data.local_struct_internal_ptr,"",@
        .globl  local_struct_internal_ptr
        .hidden  local_struct_internal_ptr
        .p2align        2
local_struct_internal_ptr:
        .int32  local_struct+4
        .size   local_struct_internal_ptr, 4
        .size   hello_str, 4

# CHECK:        - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHK64-NEXT:       - Flags:           [ IS_64 ]
# CHECK-NEXT:         Initial:         0x2
# CHECK-NEXT:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHK32-NEXT:         Type:            I32
# CHK64-NEXT:         Type:            I64
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHK32-NEXT:           Opcode:          I32_CONST
# CHK64-NEXT:           Opcode:          I64_CONST
# CHECK-NEXT:           Value:           66624
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1080
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66624

# CHECK:        - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   7
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHK32-NEXT:           Opcode:          I32_CONST
# CHK64-NEXT:           Opcode:          I64_CONST
# CHECK-NEXT:           Value:           1024
# CHECK-NEXT:         Content:         68656C6C6F0A00
# CHECK-NEXT:       - SectionOffset:   20
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHK32-NEXT:           Opcode:          I32_CONST
# CHK64-NEXT:           Opcode:          I64_CONST
# CHECK-NEXT:           Value:           1040


# RUN: wasm-ld -no-gc-sections --allow-undefined --no-entry \
# RUN:     --initial-memory=131072 --max-memory=131072 -o %t_max.wasm %t32.o \
# RUN:     %t.hello32.o
# RUN: obj2yaml %t_max.wasm | FileCheck %s -check-prefix=CHECK-MAX

# CHECK-MAX:        - Type:            MEMORY
# CHECK-MAX-NEXT:     Memories:
# CHECK-MAX-NEXT:       - Flags:           [ HAS_MAX ]
# CHECK-MAX-NEXT:         Initial:         0x2
# CHECK-MAX-NEXT:         Maximum:         0x2

# RUN: wasm-ld -no-gc-sections --allow-undefined --no-entry --shared-memory \
# RUN:     --features=atomics,bulk-memory --initial-memory=131072 \
# RUN:     --max-memory=131072 -o %t_max.wasm %t32.o %t.hello32.o
# RUN: obj2yaml %t_max.wasm | FileCheck %s -check-prefix=CHECK-SHARED

# CHECK-SHARED:        - Type:            MEMORY
# CHECK-SHARED-NEXT:     Memories:
# CHECK-SHARED-NEXT:       - Flags:           [ HAS_MAX, IS_SHARED ]
# CHECK-SHARED-NEXT:         Initial:         0x2
# CHECK-SHARED-NEXT:         Maximum:         0x2

# XUN: wasm-ld --relocatable -o %t_reloc.wasm %t32.o %t.hello32.o
# XUN: obj2yaml %t_reloc.wasm | FileCheck %s -check-prefix=RELOC

# RELOC:       - Type:            DATA
# RELOC-NEXT:     Relocations:
# RELOC-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
# RELOC-NEXT:         Index:           3
# RELOC-NEXT:         Offset:          0x24
# RELOC-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
# RELOC-NEXT:         Index:           4
# RELOC-NEXT:         Offset:          0x2D
# RELOC-NEXT:         Addend:          4
# RELOC-NEXT:     Segments:
# RELOC-NEXT:       - SectionOffset:   6
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           0
# RELOC-NEXT:         Content:         68656C6C6F0A00
# RELOC-NEXT:       - SectionOffset:   18
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           8
# RELOC-NEXT:         Content:         '01000000'
# RELOC-NEXT:       - SectionOffset:   27
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           16
# RELOC-NEXT:         Content:         '03000000'
# RELOC-NEXT:       - SectionOffset:   36
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           24
# RELOC-NEXT:         Content:         '00000000'
# RELOC-NEXT:       - SectionOffset:   45
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           28
# RELOC-NEXT:         Content:         '24000000'
# RELOC-NEXT:       - SectionOffset:   54
# RELOC-NEXT:         InitFlags:       0
# RELOC-NEXT:         Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           32
# RELOC-NEXT:         Content:         '0000000000000000'

# RELOC:          SymbolTable:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            foo
# RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# RELOC-NEXT:         Segment:         1
# RELOC-NEXT:         Size:            4
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            aligned_bar
# RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# RELOC-NEXT:         Segment:         2
# RELOC-NEXT:         Size:            4
# RELOC-NEXT:       - Index:           2
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            external_ref
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Segment:         3
# RELOC-NEXT:         Size:            4
# RELOC-NEXT:       - Index:           3
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            hello_str
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Segment:         0
# RELOC-NEXT:         Size:            7
