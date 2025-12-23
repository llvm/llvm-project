# RUN: llvm-mc -filetype=obj -mattr=+reference-types -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK -dump-input=always

# RUN: llvm-mc -filetype=obj -mattr=+reference-types -triple=wasm64-unknown-unknown %s -o %t.o
# RUN: wasm-ld --no-entry -mwasm64 -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK -dump-input=always

#      CHECK:  - Type:            TABLE
# CHECK-NEXT:    Tables:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ElemType:        EXTERNREF
# CHECK-NEXT:        Limits:
# CHECK-NEXT:          Minimum:         0x0
	.tabletype table_i32, externref, i32
table_i32:

# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        ElemType:        EXTERNREF
# CHECK-NEXT:        Limits:
# CHECK-NEXT:          Flags:         [ IS_64 ]
# CHECK-NEXT:          Minimum:         0x0
	.tabletype table_i64, externref, i64
table_i64:

	.globl	table_i32_get
table_i32_get:
    .functype table_i32_get (i32) -> (externref)
    local.get 0
    table.get table_i32
    end_function

	.globl	table_i64_get
table_i64_get:
    .functype table_i64_get (i64) -> (externref)
    local.get 0
    table.get table_i64
    end_function

    .export_name	table_i32_get, table_i32_get
    .export_name	table_i64_get, table_i64_get
