# This reproduces a bug with instrumentation crashes on internal call

# REQUIRES: system-linux,bolt-runtime,target=x86_64{{.*}}

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# Delete our BB symbols so BOLT doesn't mark them as entry points
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q -shared -fini=_fini
# RUN: llvm-bolt --instrument %t.exe --relocs -o %t.out

  .text
  .globl  _start
  .type _start, %function
  .p2align  4
_start:
  push   %rbp
  mov    %rsp,%rbp
  push   %r12
  push   %rbx
  sub    $0x120,%rsp
  mov    $0x3,%rbx
.J1:
  cmp    $0x0,%rbx
  je     .J2
  callq  .J3
  nopl   (%rax,%rax,1)
  lea    var@GOTPCREL(%rip),%rax
  retq
.J2:
  add    $0x120,%rsp
  pop    %rbx
  pop    %r12
  jmp    .J4
.J3:
  pop    %rax
  add    $0x4,%rax
  dec    %rbx
  jmp    .J1
.J4:
  pop    %rbp
  retq
  .size _start, .-_start

  .globl  _fini
  .type _fini, %function
  .p2align  4
_fini:
  hlt
  .size _fini, .-_fini

  .data
  .globl var
var:
  .quad 0xdeadbeef
