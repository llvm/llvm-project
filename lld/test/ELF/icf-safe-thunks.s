# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: ld.lld a.o -o a.so --icf=safe_thunks --print-icf-sections | FileCheck %s
# RUN: llvm-objdump a.so -d | FileCheck %s --check-prefixes=CHECK-ARM64

# CHECK: selected section a.o:(.text.func_3identical_v1_canmerge)
# CHECK: selected section a.o:(.text.func_call_thunked_1_nomerge)
# CHECK: selected section a.o:(.text.func_unique_2_canmerge)
# CHECK: selected section a.o:(.text.func_3identical_v1)

# CHECK-ARM64: func_unique_1
# CHECK-ARM64-NEXT: adrp    x8, {{.*}}
# CHECK-ARM64-NEXT: mov     w9, #0x1

# CHECK-ARM64: func_unique_2_canmerge
# CHECK-ARM64-NEXT: adrp    x8, {{.*}}
# CHECK-ARM64-NEXT: mov     w9, #0x2

# CHECK-ARM64: func_3identical_v1
# CHECK-ARM64-NEXT: adrp    x8, {{.*}}
# CHECK-ARM64-NEXT: mov     w9, #0x3

# CHECK-ARM64: func_3identical_v1_canmerge
# CHECK-ARM64-NEXT: adrp    x8, {{.*}}

# CHECK-ARM64: func_call_thunked_1_nomerge
# CHECK-ARM64-NEXT: stp     x29, x30, [sp, #-0x10]!

# CHECK-ARM64: <func_3identical_v2_canmerge>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_3identical_v1_canmerge>
# CHECK-ARM64: <func_3identical_v3_canmerge>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_3identical_v1_canmerge>
# CHECK-ARM64: <func_call_thunked_2_nomerge>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_call_thunked_1_nomerge>
# CHECK-ARM64: <func_call_thunked_2_merge>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_call_thunked_1_nomerge>
# CHECK-ARM64: <func_2identical_v1>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_unique_2_canmerge>
# CHECK-ARM64: <func_2identical_v2>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_unique_2_canmerge>
# CHECK-ARM64: <func_3identical_v2>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_3identical_v1>
# CHECK-ARM64: <func_3identical_v3>:
# CHECK-ARM64-NEXT: b       0x[[#%.6x,]] <func_3identical_v1>

;--- a.cc
#define ATTR __attribute__((noinline,used,retain)) extern "C"
typedef unsigned long long ULL;

volatile char g_val = 0;
void *volatile g_ptr = 0;

ATTR void func_unique_1() { g_val = 1; }

ATTR void func_unique_2_canmerge() { g_val = 2; }

ATTR void func_2identical_v1() { g_val = 2; }

ATTR void func_2identical_v2() { g_val = 2; }

ATTR void func_3identical_v1() { g_val = 3; }

ATTR void func_3identical_v2() { g_val = 3; }

ATTR void func_3identical_v3() { g_val = 3; }

ATTR void func_3identical_v1_canmerge() { g_val = 33; }

ATTR void func_3identical_v2_canmerge() { g_val = 33; }

ATTR void func_3identical_v3_canmerge() { g_val = 33; }

ATTR void func_call_thunked_1_nomerge() {
    func_2identical_v1();
    g_val = 77;
}

ATTR void func_call_thunked_2_nomerge() {
    func_2identical_v2();
    g_val = 77;
}

ATTR void func_call_thunked_2_merge() {
    func_2identical_v2();
    g_val = 77;
}

ATTR void call_all_funcs() {
    func_unique_1();
    func_unique_2_canmerge();
    func_2identical_v1();
    func_2identical_v2();
    func_3identical_v1();
    func_3identical_v2();
    func_3identical_v3();
    func_3identical_v1_canmerge();
    func_3identical_v2_canmerge();
    func_3identical_v3_canmerge();
}

ATTR void take_func_addr() {
    g_ptr = (void*)func_unique_1;
    g_ptr = (void*)func_2identical_v1;
    g_ptr = (void*)func_2identical_v2;
    g_ptr = (void*)func_3identical_v1;
    g_ptr = (void*)func_3identical_v2;
    g_ptr = (void*)func_3identical_v3;
}

ATTR int _start() { return 0; }

;--- gen
clang --target=aarch64-linux-gnu -O3 -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -S a.cc -o -

;--- a.s
	.file	"a.cc"
	.section	.text.func_unique_1,"axR",@progbits
	.globl	func_unique_1                   // -- Begin function func_unique_1
	.p2align	2
	.type	func_unique_1,@function
func_unique_1:                          // @func_unique_1
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #1                          // =0x1
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end0:
	.size	func_unique_1, .Lfunc_end0-func_unique_1
                                        // -- End function
	.section	.text.func_unique_2_canmerge,"axR",@progbits
	.globl	func_unique_2_canmerge          // -- Begin function func_unique_2_canmerge
	.p2align	2
	.type	func_unique_2_canmerge,@function
func_unique_2_canmerge:                 // @func_unique_2_canmerge
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #2                          // =0x2
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end1:
	.size	func_unique_2_canmerge, .Lfunc_end1-func_unique_2_canmerge
                                        // -- End function
	.section	.text.func_2identical_v1,"axR",@progbits
	.globl	func_2identical_v1              // -- Begin function func_2identical_v1
	.p2align	2
	.type	func_2identical_v1,@function
func_2identical_v1:                     // @func_2identical_v1
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #2                          // =0x2
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end2:
	.size	func_2identical_v1, .Lfunc_end2-func_2identical_v1
                                        // -- End function
	.section	.text.func_2identical_v2,"axR",@progbits
	.globl	func_2identical_v2              // -- Begin function func_2identical_v2
	.p2align	2
	.type	func_2identical_v2,@function
func_2identical_v2:                     // @func_2identical_v2
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #2                          // =0x2
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end3:
	.size	func_2identical_v2, .Lfunc_end3-func_2identical_v2
                                        // -- End function
	.section	.text.func_3identical_v1,"axR",@progbits
	.globl	func_3identical_v1              // -- Begin function func_3identical_v1
	.p2align	2
	.type	func_3identical_v1,@function
func_3identical_v1:                     // @func_3identical_v1
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #3                          // =0x3
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end4:
	.size	func_3identical_v1, .Lfunc_end4-func_3identical_v1
                                        // -- End function
	.section	.text.func_3identical_v2,"axR",@progbits
	.globl	func_3identical_v2              // -- Begin function func_3identical_v2
	.p2align	2
	.type	func_3identical_v2,@function
func_3identical_v2:                     // @func_3identical_v2
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #3                          // =0x3
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end5:
	.size	func_3identical_v2, .Lfunc_end5-func_3identical_v2
                                        // -- End function
	.section	.text.func_3identical_v3,"axR",@progbits
	.globl	func_3identical_v3              // -- Begin function func_3identical_v3
	.p2align	2
	.type	func_3identical_v3,@function
func_3identical_v3:                     // @func_3identical_v3
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #3                          // =0x3
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end6:
	.size	func_3identical_v3, .Lfunc_end6-func_3identical_v3
                                        // -- End function
	.section	.text.func_3identical_v1_canmerge,"axR",@progbits
	.globl	func_3identical_v1_canmerge     // -- Begin function func_3identical_v1_canmerge
	.p2align	2
	.type	func_3identical_v1_canmerge,@function
func_3identical_v1_canmerge:            // @func_3identical_v1_canmerge
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #33                         // =0x21
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end7:
	.size	func_3identical_v1_canmerge, .Lfunc_end7-func_3identical_v1_canmerge
                                        // -- End function
	.section	.text.func_3identical_v2_canmerge,"axR",@progbits
	.globl	func_3identical_v2_canmerge     // -- Begin function func_3identical_v2_canmerge
	.p2align	2
	.type	func_3identical_v2_canmerge,@function
func_3identical_v2_canmerge:            // @func_3identical_v2_canmerge
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #33                         // =0x21
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end8:
	.size	func_3identical_v2_canmerge, .Lfunc_end8-func_3identical_v2_canmerge
                                        // -- End function
	.section	.text.func_3identical_v3_canmerge,"axR",@progbits
	.globl	func_3identical_v3_canmerge     // -- Begin function func_3identical_v3_canmerge
	.p2align	2
	.type	func_3identical_v3_canmerge,@function
func_3identical_v3_canmerge:            // @func_3identical_v3_canmerge
// %bb.0:                               // %entry
	adrp	x8, g_val
	mov	w9, #33                         // =0x21
	strb	w9, [x8, :lo12:g_val]
	ret
.Lfunc_end9:
	.size	func_3identical_v3_canmerge, .Lfunc_end9-func_3identical_v3_canmerge
                                        // -- End function
	.section	.text.func_call_thunked_1_nomerge,"axR",@progbits
	.globl	func_call_thunked_1_nomerge     // -- Begin function func_call_thunked_1_nomerge
	.p2align	2
	.type	func_call_thunked_1_nomerge,@function
func_call_thunked_1_nomerge:            // @func_call_thunked_1_nomerge
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	bl	func_2identical_v1
	adrp	x8, g_val
	mov	w9, #77                         // =0x4d
	strb	w9, [x8, :lo12:g_val]
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	ret
.Lfunc_end10:
	.size	func_call_thunked_1_nomerge, .Lfunc_end10-func_call_thunked_1_nomerge
                                        // -- End function
	.section	.text.func_call_thunked_2_nomerge,"axR",@progbits
	.globl	func_call_thunked_2_nomerge     // -- Begin function func_call_thunked_2_nomerge
	.p2align	2
	.type	func_call_thunked_2_nomerge,@function
func_call_thunked_2_nomerge:            // @func_call_thunked_2_nomerge
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	bl	func_2identical_v2
	adrp	x8, g_val
	mov	w9, #77                         // =0x4d
	strb	w9, [x8, :lo12:g_val]
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	ret
.Lfunc_end11:
	.size	func_call_thunked_2_nomerge, .Lfunc_end11-func_call_thunked_2_nomerge
                                        // -- End function
	.section	.text.func_call_thunked_2_merge,"axR",@progbits
	.globl	func_call_thunked_2_merge       // -- Begin function func_call_thunked_2_merge
	.p2align	2
	.type	func_call_thunked_2_merge,@function
func_call_thunked_2_merge:              // @func_call_thunked_2_merge
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	bl	func_2identical_v2
	adrp	x8, g_val
	mov	w9, #77                         // =0x4d
	strb	w9, [x8, :lo12:g_val]
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	ret
.Lfunc_end12:
	.size	func_call_thunked_2_merge, .Lfunc_end12-func_call_thunked_2_merge
                                        // -- End function
	.section	.text.call_all_funcs,"axR",@progbits
	.globl	call_all_funcs                  // -- Begin function call_all_funcs
	.p2align	2
	.type	call_all_funcs,@function
call_all_funcs:                         // @call_all_funcs
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	bl	func_unique_1
	bl	func_unique_2_canmerge
	bl	func_2identical_v1
	bl	func_2identical_v2
	bl	func_3identical_v1
	bl	func_3identical_v2
	bl	func_3identical_v3
	bl	func_3identical_v1_canmerge
	bl	func_3identical_v2_canmerge
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	b	func_3identical_v3_canmerge
.Lfunc_end13:
	.size	call_all_funcs, .Lfunc_end13-call_all_funcs
                                        // -- End function
	.section	.text.take_func_addr,"axR",@progbits
	.globl	take_func_addr                  // -- Begin function take_func_addr
	.p2align	2
	.type	take_func_addr,@function
take_func_addr:                         // @take_func_addr
// %bb.0:                               // %entry
	adrp	x8, g_ptr
	adrp	x9, func_unique_1
	add	x9, x9, :lo12:func_unique_1
	str	x9, [x8, :lo12:g_ptr]
	adrp	x9, func_2identical_v1
	add	x9, x9, :lo12:func_2identical_v1
	str	x9, [x8, :lo12:g_ptr]
	adrp	x9, func_2identical_v2
	add	x9, x9, :lo12:func_2identical_v2
	str	x9, [x8, :lo12:g_ptr]
	adrp	x9, func_3identical_v1
	add	x9, x9, :lo12:func_3identical_v1
	str	x9, [x8, :lo12:g_ptr]
	adrp	x9, func_3identical_v2
	add	x9, x9, :lo12:func_3identical_v2
	str	x9, [x8, :lo12:g_ptr]
	adrp	x9, func_3identical_v3
	add	x9, x9, :lo12:func_3identical_v3
	str	x9, [x8, :lo12:g_ptr]
	ret
.Lfunc_end14:
	.size	take_func_addr, .Lfunc_end14-take_func_addr
                                        // -- End function
	.section	.text._start,"axR",@progbits
	.globl	_start                          // -- Begin function _start
	.p2align	2
	.type	_start,@function
_start:                                 // @_start
// %bb.0:                               // %entry
	mov	w0, wzr
	ret
.Lfunc_end15:
	.size	_start, .Lfunc_end15-_start
                                        // -- End function
	.type	g_val,@object                   // @g_val
	.section	.bss.g_val,"aw",@nobits
	.globl	g_val
	.p2align	2, 0x0
g_val:
	.byte	0                               // 0x0
	.size	g_val, 1

	.type	g_ptr,@object                   // @g_ptr
	.section	.bss.g_ptr,"aw",@nobits
	.globl	g_ptr
	.p2align	3, 0x0
g_ptr:
	.xword	0
	.size	g_ptr, 8

	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym func_unique_1
	.addrsig_sym func_unique_2_canmerge
	.addrsig_sym func_2identical_v1
	.addrsig_sym func_2identical_v2
	.addrsig_sym func_3identical_v1
	.addrsig_sym func_3identical_v2
	.addrsig_sym func_3identical_v3
	.addrsig_sym func_3identical_v1_canmerge
	.addrsig_sym func_3identical_v2_canmerge
	.addrsig_sym func_3identical_v3_canmerge
	.addrsig_sym func_call_thunked_1_nomerge
	.addrsig_sym func_call_thunked_2_nomerge
	.addrsig_sym func_call_thunked_2_merge
	.addrsig_sym call_all_funcs
	.addrsig_sym take_func_addr
	.addrsig_sym _start
	.addrsig_sym g_val
	.addrsig_sym g_ptr
