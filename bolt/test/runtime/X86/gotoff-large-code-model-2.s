# A variation of gotoff-large-code-model.s that accesses GOT value
# with a slightly different code sequence.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --funcs init_impls --lite \
# RUN:   -o %t.bolted
# RUN: %t.bolted | FileCheck %s

	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"Hello, world\n"
	.text
	.p2align 4
	.globl	init_impls
	.type	init_impls, @function
init_impls:
	.cfi_startproc
  push   %rbp
  mov    %rsp,%rbp
  push   %r15
  push   %rbx
  sub    $0x8,%rsp
  lea    1f(%rip),%rbx
  #  R_X86_64_GOTPC64  _GLOBAL_OFFSET_TABLE_+0x2
1: movabsq $_GLOBAL_OFFSET_TABLE_, %r11
  add    %r11,%rbx
  #  R_X86_64_GOTOFF64 .LC2
  movabs $.LC2@gotoff,%rax
  lea    (%rbx,%rax,1),%rax
  mov    %rax,%rdi
  mov    %rbx,%r15
  #  R_X86_64_PLTOFF64 puts
  movabs $puts@pltoff,%rax
  add    %rbx,%rax
  call   *%rax
  add    $0x8,%rsp
  pop    %rbx
  pop    %r15
  pop    %rbp
  retq
  .cfi_endproc
  .size init_impls, .-init_impls

  .globl main
  .type main, @function
  .p2align 4
main:
  callq init_impls
  xorq  %rax, %rax
  ret

# CHECK: Hello, world
