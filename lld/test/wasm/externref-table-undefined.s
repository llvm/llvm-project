# RUN: llvm-mc -filetype=obj -mattr=+reference-types -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.tabletype	__externref_table, externref

.globl _start
_start:
  .functype _start () -> ()
	i32.const	0
	i32.load	p
	table.get	__externref_table
	drop
	end_function



.section	.bss.p,"",@
.globl	p
.p2align	2, 0x0
p:
	.int32	0
	.size	p, 4

# CHECK:      - Type:            TABLE
# CHECK-NEXT:    Tables:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ElemType:        EXTERNREF
# CHECK-NEXT:        Limits:
# CHECK-NEXT:          Minimum:         0x0
