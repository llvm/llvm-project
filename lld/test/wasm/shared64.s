# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.o %s
# RUN: wasm-ld -mwasm64 --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.section .data.data,"",@
data:
  .p2align 2
  .int32 2
  .size data, 4

.section .data.indirect_func_external,"",@
indirect_func_external:
  .int32 func_external
.size indirect_func_external, 4

.section .data.indirect_func,"",@
indirect_func:
  .int32 foo
  .size indirect_func, 4

# Test data relocations

.section .data.data_addr,"",@
data_addr:
  .int32 data
  .size data_addr, 4

# .. against external symbols

.section .data.data_addr_external,"",@
data_addr_external:
  .int32 data_external
  .size data_addr_external, 4

# .. including addends

.section .data.extern_struct_internal_ptr,"",@
extern_struct_internal_ptr:
  .int32 extern_struct + 4
  .size extern_struct_internal_ptr, 4

# Test use of __stack_pointer

.section .text,"",@
foo:
  # %ptr = alloca i32
  # %0 = load i32, i32* @data, align 4
  # %1 = load i32 ()*, i32 ()** @indirect_func, align 4
  # call i32 %1()
  # ret i32 %0
  .functype foo () -> (i32)
  .local    i64, i32
  global.get  __stack_pointer
  i64.const 16
  i64.sub
  local.tee 0
  global.set  __stack_pointer
  global.get  __memory_base
  i64.const data@MBREL
  i64.add
  i32.load  0
  local.set 1
  global.get  indirect_func@GOT
  i32.load  0
  call_indirect  () -> (i32)
  drop
  local.get 0
  i64.const 16
  i64.add
  global.set  __stack_pointer
  local.get 1
  end_function

get_func_address:
  .functype get_func_address () -> (i32)
  global.get func_external@GOT
  end_function

get_data_address:
  .functype get_data_address () -> (i32)
  global.get  data_external@GOT
  end_function

get_local_func_address:
  # Verify that a function which is otherwise not address taken *is* added to
  # the wasm table with referenced via R_WASM_TABLE_INDEX_REL_SLEB64
  .functype get_local_func_address () -> (i64)
  global.get  __table_base
  i64.const get_func_address@TBREL
  i64.add
  end_function

.globl foo
.globl data
.globl indirect_func
.globl indirect_func_external
.globl data_addr
.globl data_addr_external
.globl extern_struct_internal_ptr
.globl get_data_address
.globl get_func_address
.globl get_local_func_address

.hidden foo
.hidden data
.hidden get_data_address
.hidden get_func_address

# Without this linking will fail because we import __stack_pointer (a mutable
# global).
# TODO(sbc): We probably want a nicer way to specify target_features section
# in assembly.
.section .custom_section.target_features,"",@
.int8 1
.int8 43
.int8 15
.ascii "mutable-globals"

.functype func_external () -> ()

# Linker-synthesized globals
.globaltype __stack_pointer, i64
.globaltype	__table_base, i64, immutable
.globaltype	__memory_base, i64, immutable

# check for dylink section at start

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink
# CHECK-NEXT:     MemorySize:      24
# CHECK-NEXT:     MemoryAlignment: 2
# CHECK-NEXT:     TableSize:       2
# CHECK-NEXT:     TableAlignment:  0
# CHECK-NEXT:     Needed:          []
# CHECK-NEXT:   - Type:            TYPE

# check for import of __table_base and __memory_base globals

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Flags:         [ IS_64 ]
# CHECK-NEXT:           Minimum:       0x1
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __indirect_function_table
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Minimum:         0x2
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __stack_pointer
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I64
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __memory_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I64
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __table_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I64
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __table_base32
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           func_external
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        1
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           indirect_func
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.func
# CHECK-NEXT:         Field:           func_external
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           data_external
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           extern_struct
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:   - Type:            FUNCTION

# CHECK:        - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            __wasm_call_ctors
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1

# check for elem segment initialized with __table_base global as offset

# CHECK:        - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Offset:
# CHECK-NEXT:           Opcode:          GLOBAL_GET
# CHECK-NEXT:           Index:           3
# CHECK-NEXT:         Functions:       [ 4, 3 ]

# check the generated code in __wasm_call_ctors and __wasm_apply_data_relocs functions
# TODO(sbc): Disassemble and verify instructions.

# CHECK:        - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            10020B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            230142047C2305360200230142087C230241016A3602002301420C7C230141006A360200230142107C2306360200230142147C230741046A3602000B

# check the data segment initialized with __memory_base global as offset

# CHECK:        - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   6
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          GLOBAL_GET
# CHECK-NEXT:           Index:           1
# CHECK-NEXT:         Content:         '020000000000000001000000000000000000000000000000'
