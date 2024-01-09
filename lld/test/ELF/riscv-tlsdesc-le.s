// RUN: llvm-mc -filetype=obj -triple=riscv64-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s
// RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=REL %s

//      CHECK: 0000000000001318 <_start>:
// CHECK-NEXT:    1318:       auipc   a0, 1
// CHECK-NEXT:    131c:       ld      a1, 1008(a0)
// CHECK-NEXT:    1320:       addi    a0, a0, 1008
// CHECK-NEXT:    1324:       jalr    t0, a1
// CHECK-NEXT:    1328:       add     a0, a0, tp
// CHECK-NEXT:    132c:       auipc   a0, 1
// CHECK-NEXT:    1330:       ld      a1, 1040(a0)
// CHECK-NEXT:    1334:       addi    a0, a0, 1040
// CHECK-NEXT:    1338:       jalr    t0, a1
// CHECK-NEXT:    133c:       add     a0, a0, tp
// CHECK-NEXT:    1340:       ret

//      REL: Relocation section '.rela.dyn' at offset 0x{{[0-9a-f]+}} contains 3 entries
//      REL: R_RISCV_TLSDESC_CALL              ffffffffffffffd4
// REL-NEXT: R_RISCV_TLSDESC_CALL              4
// REL-NEXT: R_RISCV_TLSDESC_CALL              ffffffffffffffe8

	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1"
	.file	"<stdin>"
	.globl	_start                              # -- Begin function _start
	.p2align	2
	.type	_start,@function
_start:                                     # @_start
// access local variable
.Ltlsdesc_hi0:
	auipc	a0, %tlsdesc_hi(unspecified)
	ld	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	add	a0, a0, tp

// access global variable
.Ltlsdesc_hi1:
	auipc	a0, %tlsdesc_hi(unspecified)
	ld	a1, %tlsdesc_load_lo(.Ltlsdesc_hi1)(a0)
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi1)
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi1)
	add	a0, a0, tp
	ret
.Lfunc_end0:
	.size	_start, .Lfunc_end0-_start
                                        # -- End function
	.section	".note.GNU-stack","",@progbits

	.section .tbss,"awT",@nobits
	.p2align 2
	.global v1
v1:
	.zero    4

unspecified:
	.zero    4
