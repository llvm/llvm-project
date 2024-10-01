# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.lib.o %p/Inputs/lib.s
# RUN: wasm-ld -m wasm32 --experimental-pic -shared --no-entry %t.lib.o -o %t.lib.so
# RUN: llvm-mc -filetype=obj -mattr=+reference-types -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -m wasm32 --features=mutable-globals -Bdynamic --export-table --growable-table --export-memory --export=__stack_pointer --export=__heap_base --export=__heap_end --entry=_start %t.o %t.lib.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

	.tabletype	__indirect_function_table, funcref
	.globaltype	__memory_base, i32, immutable

	.functype	f () -> ()
	.functype	_start () -> ()
	.globl	_start
	.type	_start,@function
_start:
	.functype	_start () -> ()
	global.get	__memory_base
	i32.const	f_p@MBREL
	i32.add
	i32.load	0
	call_indirect	 __indirect_function_table, () -> ()
	end_function

	.section	.data.f_p,"",@
	.globl	f_p
f_p:
	.int32	f
	.size	f_p, 4

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink.0

# CHECK:   - Type:            EXPORT
# CHECK:       - Name:            __wasm_apply_data_relocs
# CHECK-NEXT:         Kind:            FUNCTION
