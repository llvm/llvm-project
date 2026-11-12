// REQUIRES: aarch64-registered-target
// RUN: %clang -cc1as -triple arm64-apple-macos14 -filetype obj %s -o %t.o
// RUN: llvm-objdump -s --section=__debug_info %t.o \
// RUN:   | FileCheck %s --check-prefix=PLAIN
// RUN: %clang -cc1as -triple arm64-apple-macos14 -filetype obj \
// RUN:   -grelocatable-dwarf %s -o %t.reloc.o
// RUN: llvm-objdump -s --section=__debug_info %t.reloc.o \
// RUN:   | FileCheck %s --check-prefix=RELOC

// The field holds the address of "second" by default, and its offset in
// __debug_str with -grelocatable-dwarf.
// PLAIN: 0011 0a000000
// RELOC: 0000 06000000

	.text
_foo:
	ret

	.section	__DWARF,__debug_str,regular,debug
	.asciz	"first"
Lsecond:
	.asciz	"second"

	.section	__DWARF,__debug_info,regular,debug
	.long	Lsecond
