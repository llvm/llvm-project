# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-cloudabi %s -o %t.o
# RUN: ld.lld --hash-style=sysv -pie --no-relax %t.o -o %t.norelax
# RUN: llvm-readobj -r %t.norelax | FileCheck --check-prefix=NORELAX %s
# RUN: ld.lld --hash-style=sysv -pie %t.o -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck --check-prefix=RELAX %s

# If we're addressing a global relatively through the GOT and relaxations are
# disabled, we still need to emit a relocation for the entry in the GOT itself.
# NORELAX: Relocations [
# NORELAX:   Section (4) .rela.dyn {
# NORELAX:     0x{{[0-9A-F]+}} R_AARCH64_RELATIVE - 0x{{[0-9A-F]+}}
# NORELAX:   }
# NORELAX: ]

# When relaxation is enabled, the pair is relaxed and no GOT entry or dynamic
# relocation is emitted.
# RELAX: Relocations [
# RELAX-NEXT: ]

	.globl	_start
	.type	_start,@function
_start:
	adrp	x8, :got:i
	ldr	x8, [x8, :got_lo12:i]

	.type	i,@object
	.comm	i,4,4
