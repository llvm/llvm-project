# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --no-entry %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# This test defines a command with two exported functions, as well as a static
# constructor and a static destructor. Check that the exports, constructor, and
# destructor are all set up properly.

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

	.globl	some_ctor
some_ctor:
	.functype	some_ctor () -> ()
	end_function

	.globl	some_dtor
some_dtor:
	.functype	some_dtor () -> ()
	end_function

	.hidden	__cxa_atexit
	.globl	__cxa_atexit
__cxa_atexit:
	.functype	__cxa_atexit (i32, i32, i32) -> (i32)
	i32.const	0
	end_function

	.section	.text..Lcall_dtors.1,"",@
.Lcall_dtors.1:
	.functype	.Lcall_dtors.1 (i32) -> ()
	call	some_dtor
	end_function

	.section	.text..Lregister_call_dtors.1,"",@
.Lregister_call_dtors.1:
	.functype	.Lregister_call_dtors.1 () -> ()
	block
	i32.const	.Lcall_dtors.1
	i32.const	0
	i32.const	0
	call	__cxa_atexit
	i32.eqz
	br_if   	0
	unreachable
.LBB6_2:
	end_block
	end_function

	.section	.init_array.1,"",@
	.p2align	2
	.int32	some_ctor
	.int32	.Lregister_call_dtors.1
	.export_name	foo_i32, foo_i32
	.export_name	foo_f64, foo_f64

# CHECK:       - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            foo_i32
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           8
# CHECK-NEXT:      - Name:            foo_f64
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           9

# CHECK:       - Type:            CODE

# CHECK:           - Index:           8
# CHECK-NEXT:        Locals:          []
# CHECK-NEXT:        Body:            10002000200110010B
# CHECK-NEXT:      - Index:           9
# CHECK-NEXT:        Locals:          []
# CHECK-NEXT:        Body:            10002000200110020B

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __wasm_call_ctors
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            foo_i32
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            foo_f64
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            some_ctor
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        Name:            some_dtor
# CHECK-NEXT:      - Index:           5
# CHECK-NEXT:        Name:            __cxa_atexit
# CHECK-NEXT:      - Index:           6
# CHECK-NEXT:        Name:            .Lcall_dtors.1
# CHECK-NEXT:      - Index:           7
# CHECK-NEXT:        Name:            .Lregister_call_dtors.1
# CHECK-NEXT:      - Index:           8
# CHECK-NEXT:        Name:            foo_i32.command_export
# CHECK-NEXT:      - Index:           9
# CHECK-NEXT:        Name:            foo_f64.command_export
