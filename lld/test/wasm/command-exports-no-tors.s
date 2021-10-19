# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --no-entry %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Like command-exports.s, but with no ctors or dtors, so there should be no
# __wasm_call_ctors, __cxa_atexit, or wrappers.

	.globl	foo_i32
foo_i32:
	.functype	foo_i32 (i32, i32) -> (i32)
	local.get	0
	local.get	1
	i32.add
	end_function

	.globl	foo_f64
foo_f64:
	.functype	foo_f64 (f64, f64) -> (f64)
	local.get	0
	local.get	1
	f64.add
	end_function

	.export_name	foo_i32, foo_i32
	.export_name	foo_f64, foo_f64

# CHECK:       - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            foo_i32
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            foo_f64
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           1

# CHECK:       - Type:            CODE

# CHECK:           - Index:           0
# CHECK-NEXT:        Locals:          []
# CHECK-NEXT:        Body:            200020016A0B
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Locals:          []
# CHECK-NEXT:        Body:            20002001A00B

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            foo_i32
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            foo_f64
