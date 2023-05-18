// RUN: %clang_cc1 -E -x assembler-with-cpp %s -o - | FileCheck %s --strict-whitespace

.intel_syntax noprefix
.text
	.global _main
_main:
# asdf
# asdf
	mov	bogus_name, 20
	mov	rax, 5
	ret

// CHECK-LABEL: _main:
// CHECK-NEXT: {{^}} # asdf
// CHECK-NEXT: {{^}} # asdf
// CHECK-NEXT: mov bogus_name, 20
