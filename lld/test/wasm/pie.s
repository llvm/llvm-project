# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten %S/Inputs/internal_func.s -o %t.internal_func.o
# RUN: wasm-ld --no-gc-sections --experimental-pic -pie --unresolved-symbols=import-dynamic -o %t.wasm %t.o %t.internal_func.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors,__wasm_apply_data_relocs --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DISASSEM

.functype external_func () -> ()
.functype internal_func1 () -> (i32)
.functype internal_func2 () -> (i32)
.globaltype __stack_pointer, i32

.section  .data,"",@
.p2align  2
data:
  .int32  2
  .size data, 4

indirect_func:
  .int32  foo
  .size indirect_func, 4

data_addr:
  .int32  data
  .size data_addr, 4

data_addr_external:
  .int32 data_external
  .size data_addr_external, 4

.section  .text,"",@
foo:
  .functype foo () -> (i32)

  global.get data@GOT
  i32.load 0
  drop

  global.get indirect_func@GOT
  i32.load 0
  call_indirect  () -> (i32)
  drop

  global.get __stack_pointer
  end_function

get_data_address:
  .functype get_data_address () -> (i32)
  global.get data_addr_external@GOT
  end_function

get_internal_func1_address:
  .functype get_internal_func1_address () -> (i32)
  global.get internal_func1@GOT
  end_function

get_internal_func2_address:
  .functype get_internal_func2_address () -> (i32)
  global.get internal_func2@GOT
  end_function

.globl _start
_start:
  .functype _start () -> ()
  call external_func
  end_function

.section  .custom_section.target_features,"",@
.int8 2
.int8 43
.int8 7
.ascii  "atomics"
.int8 43
.int8 11
.ascii  "bulk-memory"

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink.0
# CHECK-NEXT:     MemorySize:      16
# CHECK-NEXT:     MemoryAlignment: 2
# CHECK-NEXT:     TableSize:       3
# CHECK-NEXT:     TableAlignment:  0
# CHECK-NEXT:     Needed:          []

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __indirect_function_table
# CHECK-NEXT:        Kind:            TABLE
# CHECK-NEXT:        Table:
# CHECK-NEXT:          Index:           0
# CHECK-NEXT:          ElemType:        FUNCREF
# CHECK-NEXT:          Limits:
# CHECK-NEXT:            Minimum:         0x3
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __stack_pointer
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __memory_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __table_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   false

# CHECK:        - Type:            START
# CHECK-NEXT:     StartFunction:   3

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            external_func
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            __wasm_call_ctors
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            __wasm_apply_data_relocs
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            __wasm_apply_global_relocs
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Name:            foo
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Name:            get_data_address
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Name:            get_internal_func1_address
# CHECK-NEXT:       - Index:           7
# CHECK-NEXT:         Name:            get_internal_func2_address
# CHECK-NEXT:       - Index:           8
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           9
# CHECK-NEXT:         Name:            internal_func1
# CHECK-NEXT:       - Index:           10
# CHECK-NEXT:         Name:            internal_func2
# CHECK-NEXT:     GlobalNames:

# DISASSEM-LABEL:  <__wasm_call_ctors>:
# DISASSEM-EMPTY:
# DISASSEM-NEXT:   end

# DISASSEM-LABEL:  <__wasm_apply_data_relocs>:
# DISASSEM:        end

# Run the same test with extended-const support.  When this is available
# we don't need __wasm_apply_global_relocs and instead rely on the add
# instruction in the InitExpr.  We also, therefore, do not need these globals
# to be mutable.

# RUN: wasm-ld --no-gc-sections --experimental-pic -pie --unresolved-symbols=import-dynamic --extra-features=extended-const -o %t.extended.wasm %t.o %t.internal_func.o
# RUN: obj2yaml %t.extended.wasm | FileCheck %s --check-prefix=EXTENDED-CONST

# EXTENDED-CONST-NOT: __wasm_apply_global_relocs

# EXTENDED-CONST:       - Type:            GLOBAL
# EXTENDED-CONST-NEXT:    Globals:
# EXTENDED-CONST-NEXT:      - Index:           4
# EXTENDED-CONST-NEXT:        Type:            I32
# EXTENDED-CONST-NEXT:        Mutable:         false
# EXTENDED-CONST-NEXT:        InitExpr:
# EXTENDED-CONST-NEXT:          Opcode:        GLOBAL_GET
# EXTENDED-CONST-NEXT:          Index:         1
# EXTENDED-CONST-NEXT:      - Index:           5
# EXTENDED-CONST-NEXT:        Type:            I32
# EXTENDED-CONST-NEXT:        Mutable:         false
# EXTENDED-CONST-NEXT:        InitExpr:
# EXTENDED-CONST-NEXT:          Extended:        true
# EXTENDED-CONST-NEXT:          Body:            230141046A0B
# EXTENDED-CONST-NEXT:      - Index:           6
# EXTENDED-CONST-NEXT:        Type:            I32
# EXTENDED-CONST-NEXT:        Mutable:         false
# EXTENDED-CONST-NEXT:        InitExpr:
# EXTENDED-CONST-NEXT:          Extended:        true
# This instruction sequence decodes to:
# (global.get[0x23] 0x1 i32.const[0x41] 0x0C i32.add[0x6A] end[0x0b])
# EXTENDED-CONST-NEXT:          Body:            2301410C6A0B
# EXTENDED-CONST-NEXT:      - Index:           7
# EXTENDED-CONST-NEXT:        Type:            I32
# EXTENDED-CONST-NEXT:        Mutable:         false
# EXTENDED-CONST-NEXT:        InitExpr:
# EXTENDED-CONST-NEXT:          Opcode:        GLOBAL_GET
# EXTENDED-CONST-NEXT:          Index:         2
# EXTENDED-CONST-NEXT:      - Index:           8
# EXTENDED-CONST-NEXT:        Type:            I32
# EXTENDED-CONST-NEXT:        Mutable:         false
# EXTENDED-CONST-NEXT:        InitExpr:
# EXTENDED-CONST-NEXT:          Extended:        true
# This instruction sequence decodes to:
# (global.get[0x23] 0x2 i32.const[0x41] 0x1 i32.add[0x6A] end[0x0b])
# EXTENDED-CONST-NEXT:          Body:            230241016A0B

#  EXTENDED-CONST-NOT:  - Type:            START

#      EXTENDED-CONST:    FunctionNames:
# EXTENDED-CONST-NEXT:      - Index:           0
# EXTENDED-CONST-NEXT:        Name:            external_func
# EXTENDED-CONST-NEXT:      - Index:           1
# EXTENDED-CONST-NEXT:        Name:            __wasm_call_ctors
# EXTENDED-CONST-NEXT:      - Index:           2
# EXTENDED-CONST-NEXT:        Name:            __wasm_apply_data_relocs

# Run the same test with threading support.  In this mode
# we expect __wasm_init_memory and __wasm_apply_data_relocs
# to be generated along with __wasm_start as the start
# function.

# RUN: wasm-ld --no-gc-sections --shared-memory --experimental-pic -pie --unresolved-symbols=import-dynamic -o %t.shmem.wasm %t.o %t.internal_func.o
# RUN: obj2yaml %t.shmem.wasm | FileCheck %s --check-prefix=SHMEM
# RUN: llvm-objdump --disassemble-symbols=__wasm_start --no-show-raw-insn --no-leading-addr %t.shmem.wasm | FileCheck %s --check-prefix DISASSEM-SHMEM

# SHMEM:         - Type:            START
# SHMEM-NEXT:      StartFunction:   6

# DISASSEM-SHMEM-LABEL:  <__wasm_start>:
# DISASSEM-SHMEM-EMPTY:
# DISASSEM-SHMEM-NEXT:   call 5
# DISASSEM-SHMEM-NEXT:   call 4
# DISASSEM-SHMEM-NEXT:   end

# SHMEM:         FunctionNames:
# SHMEM-NEXT:      - Index:           0
# SHMEM-NEXT:        Name:            external_func
# SHMEM-NEXT:      - Index:           1
# SHMEM-NEXT:        Name:            __wasm_call_ctors
# SHMEM-NEXT:      - Index:           2
# SHMEM-NEXT:        Name:            __wasm_init_tls
# SHMEM-NEXT:      - Index:           3
# SHMEM-NEXT:        Name:            __wasm_apply_data_relocs
# SHMEM-NEXT:      - Index:           4
# SHMEM-NEXT:        Name:            __wasm_init_memory
# SHMEM-NEXT:      - Index:           5
# SHMEM-NEXT:        Name:            __wasm_apply_global_relocs
# SHMEM-NEXT:      - Index:           6
# SHMEM-NEXT:        Name:            __wasm_start
# SHMEM-NEXT:      - Index:           7
# SHMEM-NEXT:        Name:            foo
# SHMEM-NEXT:      - Index:           8
# SHMEM-NEXT:        Name:            get_data_address
# SHMEM-NEXT:      - Index:           9
# SHMEM-NEXT:        Name:            get_internal_func1_address
# SHMEM-NEXT:      - Index:           10
# SHMEM-NEXT:        Name:            get_internal_func2_address
# SHMEM-NEXT:      - Index:           11
# SHMEM-NEXT:        Name:            _start
