# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# RUN: wasm-ld --export=__wasm_call_ctors %t.o -o %t.export.wasm
# RUN: obj2yaml %t.export.wasm | FileCheck %s -check-prefix=EXPORT

# Test that we emit wrappers and call __wasm_call_ctor when not referenced.

.globl _start
_start:
	.functype	_start () -> ()
	end_function

func1:
	.functype	func1 () -> ()
	end_function

func2:
	.functype	func2 () -> ()
	end_function

__cxa_atexit:
	.functype	__cxa_atexit (i32, i32, i32) -> (i32)
	local.get	0
	end_function

.section        .text..Lcall_dtors.1,"",@
.Lcall_dtors.1:
	.functype	.Lcall_dtors.1 (i32) -> ()
	call    	func2
	end_function

.section        .text..Lregister_call_dtors.1,"",@
.Lregister_call_dtors.1:
	.functype	.Lregister_call_dtors.1 () -> ()
	i32.const	.Lcall_dtors.1
	i32.const	0
	i32.const	__dso_handle
	call    	__cxa_atexit
	drop
	end_function

.hidden  __dso_handle
.weak    __dso_handle

.section .init_array.1,"",@
.p2align 2
.int32 func1
.int32 .Lregister_call_dtors.1

# Check that we have exactly the needed exports: `memory` because that's
# currently on by default, and `_start`, because that's the default entrypoint.

# CHECK:       - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           7

# Check the body of `_start`'s command-export wrapper.

# CHECK:       - Type:            CODE

# CHECK:           - Index:           7
# CHECK-NEXT:        Locals:          []
# CHECK-NEXT:        Body:            100010010B

# Check the symbol table to ensure all the functions are here, and that
# index 7 above refers to the function we think it does.

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __wasm_call_ctors
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            func1
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            func2
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        Name:            __cxa_atexit
# CHECK-NEXT:      - Index:           5
# CHECK-NEXT:        Name:            .Lcall_dtors.1
# CHECK-NEXT:      - Index:           6
# CHECK-NEXT:        Name:            .Lregister_call_dtors.1
# CHECK-NEXT:      - Index:           7
# CHECK-NEXT:        Name:            _start.command_export

# EXPORT: __wasm_call_ctors
# EXPORT: func1
# EXPORT: func2
# EXPORT: __cxa_atexit
