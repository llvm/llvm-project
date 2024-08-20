# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=test_relative_local_data_access,test_relative_shared_data_access --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

.globaltype __memory_base, i32, immutable

.section .data.local_data,"",@
local_data:
  .int32 1
  .size local_data, 4

.section .data.external_data,"",@
external_data:
  .int32 2
  .size external_data, 4

.section .text,"",@
test_relative_local_data_access:
    .functype test_relative_local_data_access () -> ()
    global.get  __memory_base
    i32.const local_data@MBREL
    i32.add
    i32.load 0
    drop
    end_function

# Test relative address relocation in code section for shared data

test_relative_shared_data_access:
    .functype test_relative_shared_data_access () -> ()
    global.get  __memory_base
    i32.const external_data@MBREL
    i32.add
    i32.load 0
    drop
    end_function

_start:
    .functype _start () -> ()
    call test_relative_local_data_access
    call test_relative_shared_data_access
    end_function

.globl external_data
.globl _start

# Verify that `GOT.mem` has been added for `external_data`.

# CHECK: Sections:
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            dylink.0
# CHECK-NEXT:    MemorySize:      8
# CHECK-NEXT:    MemoryAlignment: 0
# CHECK-NEXT:    TableSize:       0
# CHECK-NEXT:    TableAlignment:  0
# CHECK-NEXT:    Needed:          []
# CHECK-NEXT:  - Type:            TYPE
# CHECK-NEXT:    Signatures:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:     []
# CHECK-NEXT:  - Type:            IMPORT
# CHECK-NEXT:    Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Memory:
# CHECK-NEXT:          Minimum:         0x1
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __memory_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __table_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          GOT.mem
# CHECK-NEXT:        Field:           external_data
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   true
# CHECK-NEXT:  - Type:            FUNCTION
# CHECK-NEXT:    FunctionTypes:   [ 0, 0, 0, 0, 0 ]
# CHECK-NEXT:  - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           4
# CHECK-NEXT:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            __wasm_call_ctors
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            __wasm_apply_data_relocs
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           1
# CHECK-NEXT:      - Name:            external_data
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        Index:           3
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           4

# DIS: <test_relative_local_data_access>:
# DIS-EMPTY:
# DIS-NEXT:               	global.get	0
# DIS-NEXT:               	i32.const	0
# DIS-NEXT:               	i32.add 
# DIS-NEXT:               	i32.load	0
# DIS-NEXT:               	drop
# DIS-NEXT:               	end

# DIS: <test_relative_shared_data_access>:
# DIS-EMPTY:
# DIS-NEXT:               	global.get	0
# DIS-NEXT:               	i32.const	4
# DIS-NEXT:               	i32.add 
# DIS-NEXT:               	i32.load	0
# DIS-NEXT:               	drop
# DIS-NEXT:               	end
