# This test check that the negate-ra-state CFIs are properly emitted in case of function splitting.
# The test checks two things:
#  - we split at the correct location: to test the feature,
#		we need to split *before* the bl __cxa_throw@PLT call is made,
#		so the unwinder has to unwind from the split (cold) part.
#
#  - the BOLTed binary runs, and returns the string from foo.

# REQUIRES: system-linux,bolt-runtime

# RUN: %clangxx --target=aarch64-unknown-linux-gnu %s -o %t.exe -Wl,-q
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-eh --split-strategy=profile2 \
# RUN: --split-all-cold --print-split --print-only=_Z3foov --data=%t.fdata 2>&1 | FileCheck --check-prefix=BOLT-CHECK %s
# RUN: %t.bolt | FileCheck %s --check-prefix=RUN-CHECK

# BOLT-CHECK: -------   HOT-COLD SPLIT POINT   -------
# BOLT-CHECK-EMPTY:
# BOLT-CHECK-NEXT: .Ltmp6
# BOLT-CHECK-NEXT: Exec Count
# BOLT-CHECK-NEXT: CFI State
# BOLT-CHECK-NEXT: Predecessors:
# BOLT-CHECK-NEXT: ldr
# BOLT-CHECK-NEXT: adrp
# BOLT-CHECK-NEXT: ldr
# BOLT-CHECK-NEXT: adrp
# BOLT-CHECK-NEXT: ldr
# BOLT-CHECK-NEXT: bl      __cxa_throw@PLT

# RUN-CHECK: Exception caught: Exception from foo().

#  Source for the assembly:
#
# #include <cstdio>
# #include <stdexcept>
#
# void foo() { throw std::runtime_error("Exception from foo()."); }
#
# int main() {
#   try {
#     foo();
#   } catch (const std::exception &e) {
#     printf("Exception caught: %s\n", e.what());
#   }
#   return 0;
# }

	.text
	.section	.note.gnu.property,"a",@note
	.p2align	3, 0x0
	.word	4
	.word	16
	.word	5
	.asciz	"GNU"
	.word	3221225472
	.word	4
	.word	2
	.word	0
.Lsec_end0:
	.text
	.globl	_Z3foov                         // -- Begin function _Z3foov
	.p2align	2
	.type	_Z3foov,@function
_Z3foov:                                // @_Z3foov
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 156, DW.ref.__gxx_personality_v0
	.cfi_lsda 28, .Lexception0
// %bb.0:
	hint	#25
	.cfi_negate_ra_state
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, #16                         // =0x10
	bl	__cxa_allocate_exception
	str	x0, [sp, #8]                    // 8-byte Folded Spill
.Ltmp0:
	adrp	x1, .L.str
	add	x1, x1, :lo12:.L.str
	bl	_ZNSt13runtime_errorC1EPKc
.Ltmp1:
	b	.LBB0_1
.LBB0_1:
	ldr	x0, [sp, #8]                    // 8-byte Folded Reload
	adrp	x1, :got:_ZTISt13runtime_error
	ldr	x1, [x1, :got_lo12:_ZTISt13runtime_error]
	adrp	x2, :got:_ZNSt13runtime_errorD1Ev
	ldr	x2, [x2, :got_lo12:_ZNSt13runtime_errorD1Ev]
	bl	__cxa_throw
.LBB0_2:
.Ltmp2:
	mov	x8, x0
	ldr	x0, [sp, #8]                    // 8-byte Folded Reload
	stur	x8, [x29, #-8]
	mov	w8, w1
	stur	w8, [x29, #-12]
	bl	__cxa_free_exception
	b	.LBB0_3
.LBB0_3:
	ldur	x0, [x29, #-8]
	bl	_Unwind_Resume
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             // @LPStart Encoding = omit
	.byte	255                             // @TType Encoding = omit
	.byte	1                               // Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    // >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           //   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           // >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  //   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           //     jumps to .Ltmp2
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           // >> Call Site 3 <<
	.uleb128 .Lfunc_end0-.Ltmp1             //   Call between .Ltmp1 and .Lfunc_end0
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        // -- End function
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 156, DW.ref.__gxx_personality_v0
	.cfi_lsda 28, .Lexception1
// %bb.0:
	hint	#25
	.cfi_negate_ra_state
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	stur	wzr, [x29, #-4]
.Ltmp3:
L3:
# FDATA: 1 main #L3# 1 _Z3foov 0 0 1
	bl	_Z3foov
.Ltmp4:
	b	.LBB1_1
.LBB1_1:
	b	.LBB1_6
.LBB1_2:
.Ltmp5:
	str	x0, [sp, #16]
	mov	w8, w1
	str	w8, [sp, #12]
	b	.LBB1_3
.LBB1_3:
	ldr	w8, [sp, #12]
	subs	w8, w8, #1
	b.ne	.LBB1_9
	b	.LBB1_4
.LBB1_4:
	ldr	x0, [sp, #16]
	bl	__cxa_begin_catch
	str	x0, [sp]
	ldr	x0, [sp]
	ldr	x8, [x0]
	ldr	x8, [x8, #16]
	blr	x8
	mov	x1, x0
.Ltmp6:
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	bl	printf
.Ltmp7:
	b	.LBB1_5
.LBB1_5:
	bl	__cxa_end_catch
	b	.LBB1_6
.LBB1_6:
	mov	w0, wzr
	.cfi_def_cfa wsp, 48
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #48
	.cfi_def_cfa_offset 0
	hint	#29
	.cfi_negate_ra_state
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB1_7:
	.cfi_restore_state
.Ltmp8:
	str	x0, [sp, #16]
	mov	w8, w1
	str	w8, [sp, #12]
.Ltmp9:
	bl	__cxa_end_catch
.Ltmp10:
	b	.LBB1_8
.LBB1_8:
	b	.LBB1_9
.LBB1_9:
	ldr	x0, [sp, #16]
	bl	_Unwind_Resume
.LBB1_10:
.Ltmp11:
	bl	__clang_call_terminate
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table1:
.Lexception1:
	.byte	255                             // @LPStart Encoding = omit
	.byte	156                             // @TType Encoding = indirect pcrel sdata8
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               // Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Ltmp3-.Lfunc_begin1           // >> Call Site 1 <<
	.uleb128 .Ltmp4-.Ltmp3                  //   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin1           //     jumps to .Ltmp5
	.byte	1                               //   On action: 1
	.uleb128 .Ltmp4-.Lfunc_begin1           // >> Call Site 2 <<
	.uleb128 .Ltmp6-.Ltmp4                  //   Call between .Ltmp4 and .Ltmp6
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin1           // >> Call Site 3 <<
	.uleb128 .Ltmp7-.Ltmp6                  //   Call between .Ltmp6 and .Ltmp7
	.uleb128 .Ltmp8-.Lfunc_begin1           //     jumps to .Ltmp8
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp7-.Lfunc_begin1           // >> Call Site 4 <<
	.uleb128 .Ltmp9-.Ltmp7                  //   Call between .Ltmp7 and .Ltmp9
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin1           // >> Call Site 5 <<
	.uleb128 .Ltmp10-.Ltmp9                 //   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin1          //     jumps to .Ltmp11
	.byte	3                               //   On action: 2
	.uleb128 .Ltmp10-.Lfunc_begin1          // >> Call Site 6 <<
	.uleb128 .Lfunc_end1-.Ltmp10            //   Call between .Ltmp10 and .Lfunc_end1
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
.Lcst_end1:
	.byte	1                               // >> Action Record 1 <<
                                        //   Catch TypeInfo 1
	.byte	0                               //   No further actions
	.byte	2                               // >> Action Record 2 <<
                                        //   Catch TypeInfo 2
	.byte	0                               //   No further actions
	.p2align	2, 0x0
                                        // >> Catch TypeInfos <<
	.xword	0                               // TypeInfo 2
.Ltmp12:                                // TypeInfo 1
	.xword	.L_ZTISt9exception.DW.stub-.Ltmp12
.Lttbase0:
	.p2align	2, 0x0
                                        // -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate          // -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.p2align	2
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 // @__clang_call_terminate
	.cfi_startproc
// %bb.0:
	hint	#25
	.cfi_negate_ra_state
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	__cxa_begin_catch
	bl	_ZSt9terminatev
.Lfunc_end2:
	.size	__clang_call_terminate, .Lfunc_end2-__clang_call_terminate
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Exception from foo()."
	.size	.L.str, 22

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"Exception caught: %s\n"
	.size	.L.str.1, 22

	.data
	.p2align	3, 0x0
.L_ZTISt9exception.DW.stub:
	.xword	_ZTISt9exception
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.xword	__gxx_personality_v0
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z3foov
	.addrsig_sym __cxa_allocate_exception
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __cxa_free_exception
	.addrsig_sym __cxa_throw
	.addrsig_sym __cxa_begin_catch
	.addrsig_sym printf
	.addrsig_sym __cxa_end_catch
	.addrsig_sym __clang_call_terminate
	.addrsig_sym _ZSt9terminatev
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZTISt13runtime_error
	.addrsig_sym _ZTISt9exception
