// RUN: %clang_cc1 -S %s -o - -ffreestanding -triple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=LINUX64

#include <xmmintrin.h>
struct struct1 { int x; int y; };
void __regcall v6(int a, float b, struct struct1 c) {}

void v6_caller(){
    struct struct1 c0;
    c0.x = 0xa0a0; c0.y = 0xb0b0;
    int x= 0xf0f0, y = 0x0f0f;
    v6(x,y,c0);
}

// LINUX64-LABEL: __regcall3__v6
// LINUX64: movq	%rcx, -8(%rsp)
// LINUX64: movl	%eax, -12(%rsp)
// LINUX64: movss	%xmm0, -16(%rsp)

// LINUX64-LABEL: v6_caller
// LINUX64: movl	$41120, 16(%rsp)                # imm = 0xA0A0
// LINUX64: movl	$45232, 20(%rsp)                # imm = 0xB0B0
// LINUX64: movl	$61680, 12(%rsp)                # imm = 0xF0F0
// LINUX64: movl	$3855, 8(%rsp)                  # imm = 0xF0F
// LINUX64: movl	12(%rsp), %eax
// LINUX64: cvtsi2ssl	8(%rsp), %xmm0
// LINUX64: movq	16(%rsp), %rcx
// LINUX64: callq	.L__regcall3__v6$local


struct struct2 { int x; float y; };
void __regcall v31(int a, float b, struct struct2 c) {}

void v31_caller(){
    struct struct2 c0;
    c0.x = 0xa0a0; c0.y = 0xb0b0;
    int x= 0xf0f0, y = 0x0f0f;
    v31(x,y,c0);
}

// LINUX64: __regcall3__v31:                        # @__regcall3__v31
// LINUX64: 	movq	%rcx, -8(%rsp)
// LINUX64: 	movl	%eax, -12(%rsp)
// LINUX64: 	movss	%xmm0, -16(%rsp)
// LINUX64: v31_caller:                             # @v31_caller
// LINUX64: 	movl	$41120, 16(%rsp)                # imm = 0xA0A0
// LINUX64: 	movss	.LCPI3_0(%rip), %xmm0           # xmm0 = [4.5232E+4,0.0E+0,0.0E+0,0.0E+0]
// LINUX64: 	movss	%xmm0, 20(%rsp)
// LINUX64: 	movl	$61680, 12(%rsp)                # imm = 0xF0F0
// LINUX64: 	movl	$3855, 8(%rsp)                  # imm = 0xF0F
// LINUX64: 	movl	12(%rsp), %eax
// LINUX64: 	cvtsi2ssl	8(%rsp), %xmm0
// LINUX64: 	movq	16(%rsp), %rcx
// LINUX64: 	callq	.L__regcall3__v31$local
