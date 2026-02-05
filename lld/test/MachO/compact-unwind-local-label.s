# REQUIRES: aarch64

## Test that ld64.lld handles compact unwind entries for functions with
## temporary label names (L-prefixed symbols that are not in the symbol table).

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t.o %s
# RUN: %no-arg-lld -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -o %t %t.o
# RUN: llvm-objdump --macho --unwind-info %t | FileCheck %s --check-prefix=UNWIND

# UNWIND: Contents of __unwind_info section:
# UNWIND: Second level indices:
# UNWIND: [0]: function offset=0x{{[0-9A-Fa-f]+}}
# UNWIND: [1]: function offset=0x{{[0-9A-Fa-f]+}}

.ifdef GEN
#--- test.c
#include <stdint.h>

// Give this function a temporary-label style name so it is not emitted
// into the object symbol table on Mach-O. This forces the compact unwind
// entry to reference a section offset without a matching symbol.
__attribute__((noinline))
static int foo(int x) __asm__("Lfoo");

__attribute__((noinline))
static int foo(int x) {
    volatile int y = x + 1;
    return y;
}

int main(void) {
    return foo(1);
}
#--- gen
clang -target arm64-apple-macos11.0 -S -O2 -fno-omit-frame-pointer -o - test.c
.endif
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 15, 4
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	b	Lfoo
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function Lfoo
Lfoo:                                   ; @"\01Lfoo"
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	mov	w8, #2                          ; =0x2
	str	w8, [sp, #12]
	ldr	w0, [sp, #12]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
