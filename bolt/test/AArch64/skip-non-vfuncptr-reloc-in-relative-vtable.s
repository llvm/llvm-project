// Test the fix that BOLT should skip special handling of any non-virtual
// function pointer relocations in relative vtable.

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-gnu %s -o %t.o
// RUN: %clang %cxxflags -fuse-ld=lld %t.o -o %t.so -Wl,-q
// RUN: llvm-bolt %t.so -o %t.bolted.so

	.text
	.p2align	2
	.type	foo,@function
foo:
	.cfi_startproc
	adrp	x8, _ZTV3gooE
	add	x8, x8, :lo12:_ZTV3gooE
	ldr	x0, [x8]
	ret
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc

	.type	_fake_rtti_data,@object
	.section	.rodata.cst16._fake_rtti_data,"aMG",@progbits,16,_fake_rtti_data,comdat
	.p2align	3, 0x0
_fake_rtti_data:
	.ascii	"_FAKE_RTTI_DATA_"
	.size	_fake_rtti_data, 16

	.type	_ZTV3gooE,@object
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
_ZTV3gooE:
	.word	0
	.word	_fake_rtti_data-_ZTV3gooE-8
	.word	foo@PLT-_ZTV3gooE-8
	.size	_ZTV3gooE, 12
