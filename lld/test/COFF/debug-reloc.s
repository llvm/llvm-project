# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows-gnu %s -filetype=obj -o %t.obj

# RUN: lld-link -lldmingw -debug:dwarf -out:%t.exe -entry:mainfunc -subsystem:console %t.obj
# RUN: llvm-readobj -sections %t.exe | FileCheck %s -check-prefix SECTIONS
# RUN: llvm-readobj -coff-basereloc %t.exe | FileCheck %s -check-prefix RELOCS

# SECTIONS:         Number: 3
# SECTIONS-NEXT:    Name: .data (2E 64 61 74 61 00 00 00)
# SECTIONS-NEXT:    VirtualSize: 0x8
# SECTIONS-NEXT:    VirtualAddress: 0x3000

# RELOCS:      BaseReloc [
# RELOCS-NEXT:   Entry {
# RELOCS-NEXT:     Type: DIR64
# RELOCS-NEXT:     Address: 0x3000
# RELOCS-NEXT:   }
# RELOCS-NEXT:   Entry {
# RELOCS-NEXT:     Type: ABSOLUTE
# RELOCS-NEXT:     Address: 0x3000
# RELOCS-NEXT:   }
# RELOCS-NEXT: ]

	.text
	.def	 mainfunc;
	.scl	2;
	.type	32;
	.endef
	.globl	mainfunc
mainfunc:
.Lfunc_begin0:
	xorl	%eax, %eax
	retq

	.data
	.globl	ptr
ptr:
	.quad	mainfunc

	.section	.debug_info,"dr"
	.quad	.Lfunc_begin0
