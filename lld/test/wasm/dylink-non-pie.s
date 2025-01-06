# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.lib.o %p/Inputs/ret32.s
# RUN: wasm-ld -m wasm32 --experimental-pic -shared --no-entry %t.lib.o -o %t.lib.so
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -m wasm32 -Bdynamic %t.o %t.lib.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

	.functype	ret32 (f32) -> (i32)
	.globl	_start
_start:
	.functype	_start () -> ()
	i32.const   f_p
	drop
	end_function

	.section	.data.f_p,"",@
f_p:
	.int32	ret32
	.size	f_p, 4

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink.0

# non-pie executable doesn't import __memory_base
# CHECK:   - Type:            IMPORT
# CHECK-NOT: Field:           __memory_base

# CHECK:   - Type:            EXPORT
# CHECK:       - Name:            __wasm_apply_data_relocs
# CHECK-NEXT:         Kind:            FUNCTION

# DIS:    <__wasm_apply_data_relocs>:
# DIS-EMPTY:
# DIS-NEXT:    i32.const       1024
# DIS-NEXT:    global.get      0
# DIS-NEXT:    i32.store       0
# DIS-NEXT:    end
