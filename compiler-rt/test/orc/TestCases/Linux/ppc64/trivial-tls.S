// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic ELF TLS work by adding together TLSs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (.tbss) and non-zero-initialized
// (.tdata) sections.

	.text
	.abiversion 2
	.file	"tlstest.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	main, .Lfunc_lep0-.Lfunc_gep0
# %bb.0:                                # %entry
	mflr 0
	std 30, -16(1)                          # 8-byte Folded Spill
	stdu 1, -48(1)
	addis 3, 2, x@got@tlsgd@ha
	std 0, 64(1)
	addi 3, 3, x@got@tlsgd@l
	bl __tls_get_addr(x@tlsgd)
	nop
	lwz 30, 0(3)
	addis 3, 2, y@got@tlsgd@ha
	addi 3, 3, y@got@tlsgd@l
	bl __tls_get_addr(y@tlsgd)
	nop
	lwz 3, 0(3)
	addis 4, 2, z@got@tlsgd@ha
	add 30, 3, 30
	addi 3, 4, z@got@tlsgd@l
	bl __tls_get_addr(z@tlsgd)
	nop
	lwz 3, 0(3)
	add 3, 30, 3
	extsw 3, 3
	addi 1, 1, 48
	ld 0, 16(1)
	ld 30, -16(1)                           # 8-byte Folded Reload
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	main, .Lfunc_end0-.Lfunc_begin0
                                        # -- End function
	.type	x,@object                       # @x
	.section	.tbss,"awT",@nobits
	.globl	x
	.p2align	2, 0x0
x:
	.long	0                               # 0x0
	.size	x, 4

	.type	y,@object                       # @y
	.section	.tdata,"awT",@progbits
	.globl	y
	.p2align	2, 0x0
y:
	.long	1                               # 0x1
	.size	y, 4

	.type	z,@object                       # @z
	.globl	z
	.p2align	2, 0x0
z:
	.long	4294967295                      # 0xffffffff
	.size	z, 4
