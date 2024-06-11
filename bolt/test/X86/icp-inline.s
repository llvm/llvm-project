## This test verifies the effect of -icp-inline option: that ICP is only
## performed for call targets eligible for inlining.

## The assembly was produced from C code compiled with clang-15 -O1 -S:

# int foo(int x) { return x + 1; }
# int bar(int x) { return x*100 + 42; }
# typedef int (*const fn)(int);
# fn funcs[] = { foo, bar };
#
# int main(int argc, char *argv[]) {
#   fn func = funcs[argc];
#   return func(0);
# }

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib -pie

# Without -icp-inline option, ICP is performed
# RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 --inline-small-functions\
# RUN:   -o %t.null --lite=0 \
# RUN:   --inline-small-functions-bytes=4 --print-icp --data %t.fdata \
# RUN:   | FileCheck %s --check-prefix=CHECK-NO-ICP-INLINE
# CHECK-NO-ICP-INLINE: Binary Function "main" after indirect-call-promotion
# CHECK-NO-ICP-INLINE: callq bar
# CHECK-NO-ICP-INLINE: End of Function "main"

# With -icp-inline option, ICP is not performed (size of bar > inline threshold)
# RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 --inline-small-functions\
# RUN:   -o %t.null --lite=0 \
# RUN:   --inline-small-functions-bytes=4 --icp-inline --print-icp \
# RUN:   --data %t.fdata | FileCheck %s --check-prefix=CHECK-ICP-INLINE
# CHECK-ICP-INLINE:     Binary Function "main" after indirect-call-promotion
# CHECK-ICP-INLINE:     callq *(%rcx,%rax,8)
# CHECK-ICP-INLINE-NOT: callq bar
# CHECK-ICP-INLINE:     End of Function "main"
  .globl bar
bar:
	.cfi_startproc
	imull	$0x64, %edi, %eax
	addl	$0x2a, %eax
	retq
	.cfi_endproc
.size bar, .-bar

  .globl foo
foo:
	.cfi_startproc
	leal	0x1(%rdi), %eax
	retq
	.cfi_endproc
.size foo, .-foo

  .globl main
main:
	.cfi_startproc
  pushq %rax
  .cfi_def_cfa_offset 16
  movslq  %edi, %rax
  leaq  funcs(%rip), %rcx
  xorl  %edi, %edi
LBB00_br:
  callq *(%rcx,%rax,8)
# FDATA: 1 main #LBB00_br# 1 foo 0 0 1
# FDATA: 1 main #LBB00_br# 1 bar 0 0 2
  popq  %rcx
  .cfi_def_cfa_offset 8
  retq
	.cfi_endproc
.size main, .-main

	.data
	.globl	funcs
funcs:
	.quad	foo
	.quad	bar
