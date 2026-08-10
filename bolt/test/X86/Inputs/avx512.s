.globl use_avx512
use_avx512:
  pushq   %rbp
  movq    %rsp, %rbp
  vscalefpd       {rz-sae}, %zmm2, %zmm17, %zmm19
secondary_entry:
  popq    %rbp
  retq
  nopl    (%rax)

.globl main
main:
  pushq   %rbp
  movq    %rsp, %rbp
  subq    $16, %rsp
  movl    $0, -4(%rbp)
  callq   use_avx512
  xorl    %eax, %eax
  addq    $16, %rsp
  popq    %rbp
  retq
