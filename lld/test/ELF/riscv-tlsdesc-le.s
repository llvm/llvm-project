// RUN: llvm-mc -filetype=obj -triple=riscv64-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s
// RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=REL %s

//      CHECK: 00000000000012d8 <_start>:
// CHECK-NEXT:    12d8:       auipc   a0, 1
// CHECK-NEXT:    12dc:       ld      a1, 920(a0)
// CHECK-NEXT:    12e0:       addi    a0, a0, 920
// CHECK-NEXT:    12e4:       jalr    t0, a1
// CHECK-NEXT:    12e8:       add     a0, a0, tp
// CHECK-NEXT:    12ec:       ret

//      REL: Relocation section '.rela.dyn' at offset 0x{{[0-9a-f]+}} contains 2 entries
//      REL: R_RISCV_TLSDESC_CALL ffffffffffffffe8
// REL-NEXT: R_RISCV_TLSDESC_CALL 0

	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1"
	.file	"<stdin>"
	.globl	_start                              # -- Begin function _start
	.p2align	2
	.type	_start,@function
_start:                                     # @_start
# %bb.0:                                # %entry
.Ltlsdesc_hi0:
	auipc	a0, %tlsdesc_hi(unspecified)
	ld	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	add	a0, a0, tp
	ret
.Lfunc_end0:
	.size	_start, .Lfunc_end0-_start
                                        # -- End function
	.section	".note.GNU-stack","",@progbits

        .section .tbss,"awT",@nobits
        .p2align 2

unspecified:
        .zero    4
