# Unresolve data symbols are allowing under import-dynamic when GOT
# relocations are used
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t-dynamic.o
# RUN: wasm-ld %t-dynamic.o -o %t.wasm --unresolved-symbols=import-dynamic 2>&1 | FileCheck -check-prefix=WARN %s
# WARN: wasm-ld: warning: dynamic imports are not yet stable (--unresolved-symbols=import-dynamic)
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype undef () -> ()

.globl get_data_addr
get_data_addr:
    .functype get_data_addr () -> (i32)
    global.get undef_data@GOT
    return
    end_function

.globl get_func_addr
get_func_addr:
    .functype get_func_addr () -> (i32)
    global.get undef@GOT
    return
    end_function

.globl _start
_start:
    .functype _start () -> ()
    call undef
    call get_func_addr
    drop
    call get_data_addr
    i32.load data_ptr
    drop
    end_function

.section  .data.data_ptr,"",@
data_ptr:
  .int32  data_external+42
  .size data_ptr, 4

.size data_external, 4

#      CHECK:  - Type:            IMPORT
# CHECK-NEXT:    Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           undef
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        SigIndex:        0
# CHECK-NEXT:      - Module:          GOT.mem
# CHECK-NEXT:        Field:           undef_data
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   true
# CHECK-NEXT:      - Module:          GOT.func
# CHECK-NEXT:        Field:           undef
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   true

#      CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            undef
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            __wasm_apply_data_relocs
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            get_data_addr
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            get_func_addr
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:    GlobalNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            undef_data
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            undef
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            data_external
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            __stack_pointer
# CHECK-NEXT:    DataSegmentNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            .data
# CHECK-NEXT:...
