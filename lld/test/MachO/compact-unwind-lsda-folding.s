## Verify that the compact unwind entries for two functions with identical
## unwind information and LSDA aren't folded together; see the comment in
## UnwindInfoSectionImpl::finalize for why.

# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 -o %t/lsda.o %s
# RUN: %lld -dylib --icf=all -lSystem -lc++ -o %t/liblsda.dylib %t/lsda.o
# RUN: llvm-objdump --macho --syms --unwind-info %t/liblsda.dylib | FileCheck %s

## Check that f and g have the same unwind encoding and LSDA offset (we need to
## link with ICF above in order to get the LSDA deduplicated), and that their
## compact unwind entries aren't folded.

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x,G_ADDR:]] {{.*}} __Z1gv
# CHECK:       [[#%x,H_ADDR:]] {{.*}} __Z1hv

# CHECK-LABEL: Contents of __unwind_info section:
# CHECK:       LSDA descriptors
# CHECK-NEXT:  [0]: function offset=[[#%#.8x,G_ADDR]], LSDA offset=[[#%#x,LSDA:]]
# CHECK-NEXT:  [1]: function offset=[[#%#.8x,H_ADDR]], LSDA offset=[[#%#.8x,LSDA]]
# CHECK-NEXT:  Second level indices:
# CHECK:       [1]: function offset=[[#%#.8x,G_ADDR]], encoding[0]=[[#%#x,ENCODING:]]
# CHECK:       [2]: function offset=[[#%#.8x,H_ADDR]], encoding[0]=[[#%#.8x,ENCODING]]

## Generated from the following C++ code built with:
## clang -target x86_64-apple-macosx11.0 -S -Os -fno-inline -fomit-frame-pointer
## void f(int i) { throw i; }
## void g() { try { f(1); } catch (int) {} }
## void h() { try { f(2); } catch (int) {} }

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z1fi                          ## -- Begin function _Z1fi
__Z1fi:                                 ## @_Z1fi
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movl	%edi, %ebx
	movl	$4, %edi
	callq	___cxa_allocate_exception
	movl	%ebx, (%rax)
	movq	__ZTIi@GOTPCREL(%rip), %rsi
	movq	%rax, %rdi
	xorl	%edx, %edx
	callq	___cxa_throw
	.cfi_endproc
                                        ## -- End function
	.globl	__Z1gv                          ## -- Begin function _Z1gv
__Z1gv:                                 ## @_Z1gv
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0
	pushq	%rax
	.cfi_def_cfa_offset 16
Ltmp0:
	movl	$1, %edi
	callq	__Z1fi
Ltmp1:
	ud2
LBB1_2:                                 ## %lpad
Ltmp2:
	movq	%rax, %rdi
	callq	___cxa_begin_catch
	popq	%rax
	jmp	___cxa_end_catch                ## TAILCALL
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2, 0x0
GCC_except_table1:
Lexception0:
	.byte	255                             ## @LPStart Encoding = omit
	.byte	155                             ## @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
	.byte	1                               ## Call site Encoding = uleb128
	.uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
	.uleb128 Ltmp0-Lfunc_begin0             ## >> Call Site 1 <<
	.uleb128 Ltmp1-Ltmp0                    ##   Call between Ltmp0 and Ltmp1
	.uleb128 Ltmp2-Lfunc_begin0             ##     jumps to Ltmp2
	.byte	1                               ##   On action: 1
	.uleb128 Ltmp1-Lfunc_begin0             ## >> Call Site 2 <<
	.uleb128 Lfunc_end0-Ltmp1               ##   Call between Ltmp1 and Lfunc_end0
	.byte	0                               ##     has no landing pad
	.byte	0                               ##   On action: cleanup
Lcst_end0:
	.byte	1                               ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                               ##   No further actions
	.p2align	2, 0x0
                                        ## >> Catch TypeInfos <<
	.long	__ZTIi@GOTPCREL+4               ## TypeInfo 1
Lttbase0:
	.p2align	2, 0x0
                                        ## -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z1hv                          ## -- Begin function _Z1hv
__Z1hv:                                 ## @_Z1hv
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception1
	pushq	%rax
	.cfi_def_cfa_offset 16
Ltmp3:
	movl	$2, %edi
	callq	__Z1fi
Ltmp4:
	ud2
LBB2_2:                                 ## %lpad
Ltmp5:
	movq	%rax, %rdi
	callq	___cxa_begin_catch
	popq	%rax
	jmp	___cxa_end_catch                ## TAILCALL
Lfunc_end1:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2, 0x0
GCC_except_table2:
Lexception1:
	.byte	255                             ## @LPStart Encoding = omit
	.byte	155                             ## @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase1-Lttbaseref1
Lttbaseref1:
	.byte	1                               ## Call site Encoding = uleb128
	.uleb128 Lcst_end1-Lcst_begin1
Lcst_begin1:
	.uleb128 Ltmp3-Lfunc_begin1             ## >> Call Site 1 <<
	.uleb128 Ltmp4-Ltmp3                    ##   Call between Ltmp3 and Ltmp4
	.uleb128 Ltmp5-Lfunc_begin1             ##     jumps to Ltmp5
	.byte	1                               ##   On action: 1
	.uleb128 Ltmp4-Lfunc_begin1             ## >> Call Site 2 <<
	.uleb128 Lfunc_end1-Ltmp4               ##   Call between Ltmp4 and Lfunc_end1
	.byte	0                               ##     has no landing pad
	.byte	0                               ##   On action: cleanup
Lcst_end1:
	.byte	1                               ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                               ##   No further actions
	.p2align	2, 0x0
                                        ## >> Catch TypeInfos <<
	.long	__ZTIi@GOTPCREL+4               ## TypeInfo 1
Lttbase1:
	.p2align	2, 0x0
                                        ## -- End function
.subsections_via_symbols
