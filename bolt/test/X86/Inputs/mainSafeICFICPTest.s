# generate profile
# clang++ -O2 -fprofile-generate=. main.cpp   -c -o mainProf.o
# PROF=test.profdata
# clang++ -m64  -fprofile-use=$PROF \
#   -mllvm -disable-icp=true -mllvm -print-after-all \
#   -g0 -flto=thin -fwhole-program-vtables -fno-split-lto-unit -O2 \
#   -fdebug-types-section \
#   main.cpp -c -o mainProfLTO.bc
# PASS='pgo-icall-prom'
# clang++ -m64  -fprofile-use=$PROF \
#   -O3 -Rpass=$PASS \
#   -mllvm -print-before=$PASS \
#   -mllvm -print-after=$PASS \
#   -mllvm -filter-print-funcs=main \
#   -mllvm -debug-only=$PASS \
#   -x ir \
#   mainProfLTO.bc -c -o mainProfFinal.o

# class Base {
# public:
#   virtual int func(int a, int b) const = 0;
#
#   virtual ~Base() {};
# };
#
# //namespace {
# class Derived2 : public Base {
#   int c = 5;
# public:
#   __attribute__((noinline)) int func(int a, int b)const override { return a * (a - b) + this->c; }
#
#   ~Derived2() {}
# };
#
# class Derived3 : public Base {
#   int c = 500;
# public:
#   __attribute__((noinline)) int func(int a, int b) const override { return a * (a - b) + this->c; }
#   ~Derived3() {}
# };
# //} // namespace//
#
# __attribute__((noinline)) Base *createType(int a) {
#     Base *base = nullptr;
#     if (a == 4)
#       base = new Derived2();
#     else
#       base = new Derived3();
#     return base;
# }
#
# extern int returnFive();
# extern int returnFourOrFive(int val);
# int main(int argc, char **argv) {
#   int sum = 0;
#   int a = returnFourOrFive(argc);
#   int b = returnFive();
#   Base *ptr = createType(a);
#   Base *ptr2 = createType(b);
#   sum += ptr->func(b, a) + ptr2->func(b, a);
#   return 0;
# }
  .text
	.file	"main.cpp"
	.section	.text.hot.,"ax",@progbits
	.globl	_Z10createTypei                 # -- Begin function _Z10createTypei
	.p2align	4, 0x90
	.type	_Z10createTypei,@function
_Z10createTypei:                        # @_Z10createTypei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movl	%edi, %ebx
	movl	$16, %edi
	callq	_Znwm@PLT
	cmpl	$4, %ebx
	xorps	%xmm0, %xmm0
	leaq	_ZTV8Derived2+16(%rip), %rcx
	leaq	_ZTV8Derived3+16(%rip), %rdx
	cmoveq	%rcx, %rdx
	movl	$5, %ecx
	movl	$500, %esi                      # imm = 0x1F4
	cmovel	%ecx, %esi
	movaps	%xmm0, (%rax)
	movq	%rdx, (%rax)
	movl	%esi, 8(%rax)
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	_Z10createTypei, .Lfunc_end0-_Z10createTypei
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	callq	_Z16returnFourOrFivei@PLT
	movl	%eax, %ebx
	callq	_Z10returnFivev@PLT
	movl	%eax, %ebp
	movl	%ebx, %edi
	callq	_Z10createTypei
	movq	%rax, %r15
	movl	%ebp, %edi
	callq	_Z10createTypei
	movq	%rax, %r14
	movq	(%r15), %rax
	movq	(%rax), %rax
	leaq	_ZNK8Derived24funcEii(%rip), %rcx
	movq	%r15, %rdi
	movl	%ebp, %esi
	movl	%ebx, %edx
	cmpq	%rcx, %rax
	jne	.LBB1_2
# %bb.1:                                # %if.true.direct_targ
	callq	_ZNK8Derived24funcEii
.LBB1_3:                                # %if.end.icp
	movq	(%r14), %rax
	movq	(%rax), %rax
	leaq	_ZNK8Derived34funcEii(%rip), %rcx
	movq	%r14, %rdi
	movl	%ebp, %esi
	movl	%ebx, %edx
	cmpq	%rcx, %rax
	jne	.LBB1_5
# %bb.4:                                # %if.true.direct_targ1
	callq	_ZNK8Derived34funcEii
.LBB1_6:                                # %if.end.icp3
	xorl	%eax, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB1_2:                                # %if.false.orig_indirect
	.cfi_def_cfa_offset 48
	callq	*%rax
	jmp	.LBB1_3
.LBB1_5:                                # %if.false.orig_indirect2
	callq	*%rax
	jmp	.LBB1_6
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.text.hot._ZNK8Derived24funcEii,"axG",@progbits,_ZNK8Derived24funcEii,comdat
	.weak	_ZNK8Derived24funcEii           # -- Begin function _ZNK8Derived24funcEii
	.p2align	4, 0x90
	.type	_ZNK8Derived24funcEii,@function
_ZNK8Derived24funcEii:                  # @_ZNK8Derived24funcEii
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%esi, %eax
	subl	%edx, %eax
	imull	%esi, %eax
	addl	8(%rdi), %eax
	retq
.Lfunc_end2:
	.size	_ZNK8Derived24funcEii, .Lfunc_end2-_ZNK8Derived24funcEii
	.cfi_endproc
                                        # -- End function
	.section	.text.unlikely._ZN8Derived2D0Ev,"axG",@progbits,_ZN8Derived2D0Ev,comdat
	.weak	_ZN8Derived2D0Ev                # -- Begin function _ZN8Derived2D0Ev
	.p2align	4, 0x90
	.type	_ZN8Derived2D0Ev,@function
_ZN8Derived2D0Ev:                       # @_ZN8Derived2D0Ev
	.cfi_startproc
# %bb.0:                                # %entry
	movl	$16, %esi
	jmp	_ZdlPvm@PLT                     # TAILCALL
.Lfunc_end3:
	.size	_ZN8Derived2D0Ev, .Lfunc_end3-_ZN8Derived2D0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text.hot._ZNK8Derived34funcEii,"axG",@progbits,_ZNK8Derived34funcEii,comdat
	.weak	_ZNK8Derived34funcEii           # -- Begin function _ZNK8Derived34funcEii
	.p2align	4, 0x90
	.type	_ZNK8Derived34funcEii,@function
_ZNK8Derived34funcEii:                  # @_ZNK8Derived34funcEii
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%esi, %eax
	subl	%edx, %eax
	imull	%esi, %eax
	addl	8(%rdi), %eax
	retq
.Lfunc_end4:
	.size	_ZNK8Derived34funcEii, .Lfunc_end4-_ZNK8Derived34funcEii
	.cfi_endproc
                                        # -- End function
	.section	.text.unlikely._ZN4BaseD2Ev,"axG",@progbits,_ZN4BaseD2Ev,comdat
	.weak	_ZN4BaseD2Ev                    # -- Begin function _ZN4BaseD2Ev
	.p2align	4, 0x90
	.type	_ZN4BaseD2Ev,@function
_ZN4BaseD2Ev:                           # @_ZN4BaseD2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	retq
.Lfunc_end5:
	.size	_ZN4BaseD2Ev, .Lfunc_end5-_ZN4BaseD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text.unlikely._ZN8Derived3D0Ev,"axG",@progbits,_ZN8Derived3D0Ev,comdat
	.weak	_ZN8Derived3D0Ev                # -- Begin function _ZN8Derived3D0Ev
	.p2align	4, 0x90
	.type	_ZN8Derived3D0Ev,@function
_ZN8Derived3D0Ev:                       # @_ZN8Derived3D0Ev
	.cfi_startproc
# %bb.0:                                # %entry
	movl	$16, %esi
	jmp	_ZdlPvm@PLT                     # TAILCALL
.Lfunc_end6:
	.size	_ZN8Derived3D0Ev, .Lfunc_end6-_ZN8Derived3D0Ev
	.cfi_endproc
                                        # -- End function
	.type	_ZTV8Derived2,@object           # @_ZTV8Derived2
	.section	.data.rel.ro._ZTV8Derived2,"awG",@progbits,_ZTV8Derived2,comdat
	.weak	_ZTV8Derived2
	.p2align	3, 0x0
_ZTV8Derived2:
	.quad	0
	.quad	_ZTI8Derived2
	.quad	_ZNK8Derived24funcEii
	.quad	_ZN4BaseD2Ev
	.quad	_ZN8Derived2D0Ev
	.size	_ZTV8Derived2, 40

	.type	_ZTS8Derived2,@object           # @_ZTS8Derived2
	.section	.rodata._ZTS8Derived2,"aG",@progbits,_ZTS8Derived2,comdat
	.weak	_ZTS8Derived2
_ZTS8Derived2:
	.asciz	"8Derived2"
	.size	_ZTS8Derived2, 10

	.type	_ZTS4Base,@object               # @_ZTS4Base
	.section	.rodata._ZTS4Base,"aG",@progbits,_ZTS4Base,comdat
	.weak	_ZTS4Base
_ZTS4Base:
	.asciz	"4Base"
	.size	_ZTS4Base, 6

	.type	_ZTI4Base,@object               # @_ZTI4Base
	.section	.data.rel.ro._ZTI4Base,"awG",@progbits,_ZTI4Base,comdat
	.weak	_ZTI4Base
	.p2align	3, 0x0
_ZTI4Base:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4Base
	.size	_ZTI4Base, 16

	.type	_ZTI8Derived2,@object           # @_ZTI8Derived2
	.section	.data.rel.ro._ZTI8Derived2,"awG",@progbits,_ZTI8Derived2,comdat
	.weak	_ZTI8Derived2
	.p2align	3, 0x0
_ZTI8Derived2:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTS8Derived2
	.quad	_ZTI4Base
	.size	_ZTI8Derived2, 24

	.type	_ZTV8Derived3,@object           # @_ZTV8Derived3
	.section	.data.rel.ro._ZTV8Derived3,"awG",@progbits,_ZTV8Derived3,comdat
	.weak	_ZTV8Derived3
	.p2align	3, 0x0
_ZTV8Derived3:
	.quad	0
	.quad	_ZTI8Derived3
	.quad	_ZNK8Derived34funcEii
	.quad	_ZN4BaseD2Ev
	.quad	_ZN8Derived3D0Ev
	.size	_ZTV8Derived3, 40

	.type	_ZTS8Derived3,@object           # @_ZTS8Derived3
	.section	.rodata._ZTS8Derived3,"aG",@progbits,_ZTS8Derived3,comdat
	.weak	_ZTS8Derived3
_ZTS8Derived3:
	.asciz	"8Derived3"
	.size	_ZTS8Derived3, 10

	.type	_ZTI8Derived3,@object           # @_ZTI8Derived3
	.section	.data.rel.ro._ZTI8Derived3,"awG",@progbits,_ZTI8Derived3,comdat
	.weak	_ZTI8Derived3
	.p2align	3, 0x0
_ZTI8Derived3:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTS8Derived3
	.quad	_ZTI4Base
	.size	_ZTI8Derived3, 24

	.cg_profile _Z10createTypei, _Znwm, 2
	.cg_profile main, _Z16returnFourOrFivei, 1
	.cg_profile main, _Z10returnFivev, 1
	.cg_profile main, _Z10createTypei, 2
	.cg_profile main, _ZNK8Derived24funcEii, 1
	.cg_profile main, _ZNK8Derived34funcEii, 1
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _ZTVN10__cxxabiv120__si_class_type_infoE
	.addrsig_sym _ZTS8Derived2
	.addrsig_sym _ZTVN10__cxxabiv117__class_type_infoE
	.addrsig_sym _ZTS4Base
	.addrsig_sym _ZTI4Base
	.addrsig_sym _ZTI8Derived2
	.addrsig_sym _ZTS8Derived3
	.addrsig_sym _ZTI8Derived3
