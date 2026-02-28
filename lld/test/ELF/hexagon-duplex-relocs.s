# REQUIRES: hexagon
# RUN: rm -rf %t.dir && split-file %s %t.dir
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf \
# RUN:   %t.dir/ext.s -o %t-ext.o
# RUN: ld.lld -shared %t-ext.o -soname ext -o %t-ext.so
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf \
# RUN:   %t.dir/static.s -o %t-static.o
# RUN: llvm-readobj -r %t-static.o | FileCheck --check-prefix=RELOC-S %s
# RUN: ld.lld %t-static.o -o %t-static
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t-static | \
# RUN:   FileCheck --check-prefix=STATIC %s
# RUN: llvm-mc -mno-fixup -filetype=obj -triple=hexagon-unknown-elf \
# RUN:   %t.dir/shared.s -o %t-shared.o
# RUN: llvm-readobj -r %t-shared.o | FileCheck --check-prefix=RELOC-D %s
# RUN: ld.lld -shared %t-shared.o %t-ext.so -o %t-shared.so
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t-shared.so | \
# RUN:   FileCheck --check-prefix=SHARED %s
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf \
# RUN:   %t.dir/ie.s -o %t-ie.o
# RUN: llvm-readobj -r %t-ie.o | FileCheck --check-prefix=RELOC-IE %s
# RUN: ld.lld %t-ie.o -o %t-ie
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t-ie | \
# RUN:   FileCheck --check-prefix=IE %s

## Test that duplex instructions (parse bits [15:14] == 0) get the correct
## relocation bit-mask in findMaskR6, findMaskR8, findMaskR11, findMaskR16.
## Each test pairs a non-duplex reference with a duplex instruction using the
## same symbol; both must resolve to the same value.

#--- ext.s
.global bar
bar:
	jumpr r31

#--- static.s
	.globl	_start, target
	.type	_start, @function
_start:

# R_HEX_6_X (findMaskR6, duplex SA1_seti)
# RELOC-S:      0x0 R_HEX_32_6_X target 0x0
# RELOC-S-NEXT: 0x4 R_HEX_16_X target 0x0
# STATIC:      { immext(#0x20140)
# STATIC-NEXT:   r0 = ##0x20168 }
	r0 = ##target
# RELOC-S-NEXT: 0x8 R_HEX_32_6_X target 0x0
# RELOC-S-NEXT: 0xC R_HEX_6_X target 0x0
# STATIC-NEXT: { immext(#0x20140)
# STATIC-NEXT:   r0 = ##0x20168; r2 = r16 }
	{ r0 = ##target; r2 = r16 }

# R_HEX_8_X (findMaskR8, duplex SA1_addi)
# RELOC-S-NEXT: 0x10 R_HEX_32_6_X target 0x0
# RELOC-S-NEXT: 0x14 R_HEX_16_X target 0x0
# STATIC-NEXT: { immext(#0x20140)
# STATIC-NEXT:   r0 = add(r0,##0x20168) }
	r0 = add(r0, ##target)
# RELOC-S-NEXT: 0x18 R_HEX_32_6_X target 0x0
# RELOC-S-NEXT: 0x1C R_HEX_8_X target 0x0
# STATIC-NEXT: { immext(#0x20140)
# STATIC-NEXT:   r0 = add(r0,##0x20168); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##target)
	  memw(r1+#0) = r2 }

# R_HEX_TPREL_16_X (findMaskR16, duplex SA1_seti)
# RELOC-S-NEXT: 0x20 R_HEX_TPREL_32_6_X tls_small 0x0
# RELOC-S-NEXT: 0x24 R_HEX_TPREL_16_X tls_small 0x0
# STATIC-NEXT: { immext(#0xfffbffc0)
# STATIC-NEXT:   r0 = ##-0x40008 }
	r0 = ##tls_small@TPREL
# RELOC-S-NEXT: 0x28 R_HEX_TPREL_32_6_X tls_small 0x0
# RELOC-S-NEXT: 0x2C R_HEX_TPREL_16_X tls_small 0x0
# STATIC-NEXT: { immext(#0xfffbffc0)
# STATIC-NEXT:   r0 = ##0xfffbfff8; r2 = r16 }
	{ r0 = ##tls_small@TPREL; r2 = r16 }

# R_HEX_TPREL_11_X (findMaskR11, duplex SA1_addi)
# RELOC-S-NEXT: 0x30 R_HEX_TPREL_32_6_X tls_small 0x0
# RELOC-S-NEXT: 0x34 R_HEX_TPREL_16_X tls_small 0x0
# STATIC-NEXT: { immext(#0xfffbffc0)
# STATIC-NEXT:   r0 = add(r0,##-0x40008) }
	r0 = add(r0, ##tls_small@TPREL)
# RELOC-S-NEXT: 0x38 R_HEX_TPREL_32_6_X tls_small 0x0
# RELOC-S-NEXT: 0x3C R_HEX_TPREL_11_X tls_small 0x0
# STATIC-NEXT: { immext(#0xfffbffc0)
# STATIC-NEXT:   r0 = add(r0,##-0x40008); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##tls_small@TPREL)
	  memw(r1+#0) = r2 }

# R_HEX_TPREL_11_X with large TLS offset (findMaskR11)
# RELOC-S-NEXT: 0x40 R_HEX_TPREL_32_6_X tls_big 0x0
# RELOC-S-NEXT: 0x44 R_HEX_TPREL_11_X tls_big 0x0
# STATIC-NEXT: { immext(#0xffffffc0)
# STATIC-NEXT:   r2 = memw(r2+##-0x4) }
	r2 = memw(r2+##tls_big@TPREL)
# RELOC-S-NEXT: 0x48 R_HEX_TPREL_32_6_X tls_big 0x0
# RELOC-S-NEXT: 0x4C R_HEX_TPREL_11_X tls_big 0x0
# STATIC-NEXT: { immext(#0xffffffc0)
# STATIC-NEXT:   r2 = add(r2,##-0x4); memw(r3+#0x0) = #0 }
	{ r2 = add(r2, ##tls_big@TPREL)
	  memw(r3+#0) = #0 }

	jumpr r31

target:
	nop
	jumpr r31

.section .tdata,"awT",@progbits
.globl	tls_small
.p2align 2
tls_small:
	.word	1
	.size	tls_small, 4

.section .tbss,"awT",@nobits
.p2align 2
.space	0x40000
.globl	tls_big
tls_big:
	.space	4
	.size	tls_big, 4

#--- shared.s
.global _start
_start:

# R_HEX_GOT_16_X (findMaskR16, duplex SA1_addi)
# RELOC-D:      0x0 R_HEX_GOT_32_6_X bar 0x0
# RELOC-D-NEXT: 0x4 R_HEX_GOT_16_X bar 0x0
# SHARED:      { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r1,##-0x10010) }
	r0 = add(r1, ##bar@GOT)
# RELOC-D-NEXT: 0x8 R_HEX_GOT_32_6_X bar 0x0
# RELOC-D-NEXT: 0xC R_HEX_GOT_16_X bar 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r0,##-0x10010); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##bar@GOT)
	  memw(r1+#0) = r2 }

# R_HEX_GOT_11_X (findMaskR11, duplex SA1_seti)
# RELOC-D-NEXT: 0x10 R_HEX_GOT_32_6_X bar 0x0
# RELOC-D-NEXT: 0x14 R_HEX_GOT_11_X bar 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = ##0xfffefff0; r2 = r16 }
	{ r0 = ##bar@GOT; r2 = r16 }

# R_HEX_GOTREL_11_X (findMaskR11, duplex SA1_addi)
# RELOC-D-NEXT: 0x18 R_HEX_GOTREL_32_6_X .text 0x28
# RELOC-D-NEXT: 0x1C R_HEX_GOTREL_16_X .text 0x28
# SHARED-NEXT: { immext(#0xfffdff40)
# SHARED-NEXT:   r0 = add(r1,##-0x20098) }
	r0 = add(r1, ##.Lgotrel_pc@GOTREL)
# RELOC-D-NEXT: 0x20 R_HEX_GOTREL_32_6_X .text 0x28
# RELOC-D-NEXT: 0x24 R_HEX_GOTREL_11_X .text 0x28
# SHARED-NEXT: { immext(#0xfffdff40)
# SHARED-NEXT:   r0 = add(r0,##-0x20098); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##.Lgotrel_pc@GOTREL)
	  memw(r1+#0) = r2 }
.Lgotrel_pc:

# R_HEX_GD_GOT_11_X (findMaskR11, duplex SA1_addi)
# RELOC-D-NEXT: 0x28 R_HEX_GD_GOT_32_6_X tls_a 0x0
# RELOC-D-NEXT: 0x2C R_HEX_GD_GOT_16_X tls_a 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r1,##-0x1000c) }
	r0 = add(r1, ##tls_a@GDGOT)
# RELOC-D-NEXT: 0x30 R_HEX_GD_GOT_32_6_X tls_a 0x0
# RELOC-D-NEXT: 0x34 R_HEX_GD_GOT_11_X tls_a 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r0,##-0x1000c); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##tls_a@GDGOT)
	  memw(r1+#0) = r2 }

# R_HEX_IE_GOT_11_X (findMaskR11, duplex SA1_addi)
# RELOC-D-NEXT: 0x38 R_HEX_IE_GOT_32_6_X tls_a 0x0
# RELOC-D-NEXT: 0x3C R_HEX_IE_GOT_16_X tls_a 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r1,##-0x10004) }
	r0 = add(r1, ##tls_a@IEGOT)
# RELOC-D-NEXT: 0x40 R_HEX_IE_GOT_32_6_X tls_a 0x0
# RELOC-D-NEXT: 0x44 R_HEX_IE_GOT_11_X tls_a 0x0
# SHARED-NEXT: { immext(#0xfffeffc0)
# SHARED-NEXT:   r0 = add(r0,##-0x10004); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##tls_a@IEGOT)
	  memw(r1+#0) = r2 }

	jumpr r31

.section .tdata,"awT",@progbits
.globl	tls_a
tls_a:
	.word	1

#--- ie.s
	.globl	_start
	.type	_start, @function
_start:

# R_HEX_IE_16_X (findMaskR16, duplex SA1_seti)
# RELOC-IE:      0x0 R_HEX_IE_32_6_X ie_var 0x0
# RELOC-IE-NEXT: 0x4 R_HEX_IE_16_X ie_var 0x0
# IE:      { immext(#0x30140)
# IE-NEXT:   r0 = memw(##0x3015c) }
	r0 = memw(##ie_var@IE)
# RELOC-IE-NEXT: 0x8 R_HEX_IE_32_6_X ie_var 0x0
# RELOC-IE-NEXT: 0xC R_HEX_IE_16_X ie_var 0x0
# IE-NEXT: { immext(#0x30140)
# IE-NEXT:   r0 = ##0x3015c; r2 = r16 }
	{ r0 = ##ie_var@IE; r2 = r16 }

# R_HEX_IE_GOT_11_X (findMaskR11, duplex SA1_addi)
# RELOC-IE-NEXT: 0x10 R_HEX_IE_GOT_32_6_X ie_var 0x0
# RELOC-IE-NEXT: 0x14 R_HEX_IE_GOT_16_X ie_var 0x0
# IE-NEXT: { immext(#0xfffeffc0)
# IE-NEXT:   r0 = add(r1,##-0x10004) }
	r0 = add(r1, ##ie_var@IEGOT)
# RELOC-IE-NEXT: 0x18 R_HEX_IE_GOT_32_6_X ie_var 0x0
# RELOC-IE-NEXT: 0x1C R_HEX_IE_GOT_11_X ie_var 0x0
# IE-NEXT: { immext(#0xfffeffc0)
# IE-NEXT:   r0 = add(r0,##-0x10004); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##ie_var@IEGOT)
	  memw(r1+#0) = r2 }

	jumpr r31

.section .tdata,"awT",@progbits
.globl	ie_var
.p2align 2
ie_var:
	.word	1
	.size	ie_var, 4
