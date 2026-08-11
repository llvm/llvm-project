# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.o %s
# RUN: wasm-ld -mwasm64 --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=test_indirect_local_func_call,test_indirect_shared_func_call --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

.globaltype	__table_base, i64, immutable

test_indirect_local_func_call:
    .functype test_indirect_local_func_call () -> ()
    global.get  __table_base
    i64.const local_func@TBREL
    i64.add
    i32.wrap_i64 # Remove when table64 is supported
    call_indirect () -> ()
    end_function

# Test table index relocation in code section for shared functions

test_indirect_shared_func_call:
    .functype test_indirect_shared_func_call () -> ()
    i32.const 12345
    global.get  __table_base
    i64.const shared_func@TBREL
    i64.add
    i32.wrap_i64 # Remove when table64 is supported
    call_indirect (i32) -> (i32)
    drop
    end_function

local_func:
    .functype local_func () -> ()
    end_function

.globl shared_func
shared_func:
    .functype shared_func (i32) -> (i32)
    local.get 0
    end_function

.globl _start
_start:
    .functype _start () -> ()
    call test_indirect_local_func_call
    call test_indirect_shared_func_call
    end_function

# CHECK:      Sections:
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            dylink.0
# CHECK-NEXT:    MemorySize:      0
# CHECK-NEXT:    MemoryAlignment: 0
# CHECK-NEXT:    TableSize:       2
# CHECK-NEXT:    TableAlignment:  0
# CHECK-NEXT:    Needed:          []
# CHECK-NEXT:  - Type:            TYPE

# CHECK:  - Type:            IMPORT
# CHECK-NEXT:    Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Memory:
# CHECK-NEXT:          Flags:           [ IS_64 ]
# CHECK-NEXT:          Minimum:         0x0
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __indirect_function_table
# CHECK-NEXT:        Kind:            TABLE
# CHECK-NEXT:        Table:
# CHECK-NEXT:          Index:           0
# CHECK-NEXT:          ElemType:        FUNCREF
# CHECK-NEXT:          Limits:
# CHECK-NEXT:            Flags:           [ IS_64 ]
# CHECK-NEXT:            Minimum:         0x2
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __memory_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I64
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __table_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I64
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          GOT.func
# CHECK-NEXT:        Field:           shared_func
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I64
# CHECK-NEXT:        GlobalMutable:   true
# CHECK-NEXT:  - Type:            FUNCTION
# CHECK-NEXT:    FunctionTypes:   [ 0, 0, 0, 0, 0, 1, 0 ]
# CHECK-NEXT:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            __wasm_call_ctors
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            __wasm_apply_data_relocs
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           1
# CHECK-NEXT:      - Name:            shared_func
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           5
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           6
# CHECK-NEXT:  - Type:            ELEM
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - Offset:
# CHECK-NEXT:          Opcode:          GLOBAL_GET
# CHECK-NEXT:          Index:           1
# CHECK-NEXT:        Functions:       [ 3, 5 ]


# DIS: <test_indirect_local_func_call>:
# DIS-EMPTY:
# DIS-NEXT:               	global.get	1
# DIS-NEXT:               	i64.const	0
# DIS-NEXT:               	i64.add 
# DIS-NEXT:               	i32.wrap_i64 
# DIS-NEXT:               	call_indirect	 0
# DIS-NEXT:               	end

# DIS: <test_indirect_shared_func_call>:
# DIS-EMPTY:
# DIS-NEXT:               	i32.const	12345
# DIS-NEXT:               	global.get	1
# DIS-NEXT:               	i64.const	1
# DIS-NEXT:               	i64.add 
# DIS-NEXT:               	i32.wrap_i64 
# DIS-NEXT:               	call_indirect	 1
# DIS-NEXT:               	drop
# DIS-NEXT:               	end
