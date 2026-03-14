// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -r -S --hex-dump=.debug_info %t.so | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     R_X86_64_DTPMOD64 - 0x0
// CHECK-NEXT:     R_X86_64_DTPMOD64 - 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     R_X86_64_JUMP_SLOT __tls_get_addr 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:      Hex dump of section '.debug_info':
// CHECK-NEXT: 0x00000000 04000000 00000000 00000000 00000000

	data16
	leaq	bar@TLSGD(%rip), %rdi
	data16
	data16
	rex64
	callq	__tls_get_addr@PLT

	leaq	bar@TLSLD(%rip), %rdi
	callq	__tls_get_addr@PLT
	leaq	bar@DTPOFF(%rax), %rax

	.section	.tdata,"awT",@progbits
bar:
	.long	42
foo:
	.long	43

	.section	.debug_info,"",@progbits
	.quad	foo@DTPOFF
	.quad	bar@DTPOFF
