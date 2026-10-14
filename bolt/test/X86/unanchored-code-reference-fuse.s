## Check that, when all functions are rewritten, BOLT keeps ambiguous
## references valid by emitting the functions around them unoptimized and
## back-to-back with the original bytes between them. See
## unanchored-code-reference.s for the detection of such references.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --nostdlib -e _start
## List the second function of every pair first, so that reordering would
## separate the pairs. Leave ret_next out, so that it would go to a different
## section than ret_prev.
# RUN: echo user > %t.order
# RUN: echo self_next >> %t.order
# RUN: echo named_next >> %t.order
# RUN: echo end_next >> %t.order
# RUN: echo pad_next >> %t.order
# RUN: echo mid_next >> %t.order
# RUN: echo pad_prev >> %t.order
# RUN: echo ret_prev >> %t.order
# RUN: echo mid_prev >> %t.order
# RUN: echo head >> %t.order
# RUN: echo self_prev >> %t.order
# RUN: echo named_prev >> %t.order
# RUN: echo end_prev >> %t.order
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs --lite=0 --strict \
# RUN:   --reorder-functions=user --function-order=%t.order 2>&1 | FileCheck %s
# RUN: llvm-nm %t.bolt > %t.out
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt >> %t.out
# RUN: FileCheck %s --check-prefix=CHECK-OUT --input-file=%t.out

## A function referencing itself is not reported.
# CHECK-NOT: self_prev
# CHECK: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}}: in padding 9 bytes after the end of pad_prev/1 and 1 bytes before pad_next/1; freezing and fusing pad_prev/1 and pad_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}}: inside ret_prev/1 at offset 0x5, 1 bytes before ret_next/1; freezing and fusing ret_prev/1 and ret_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}}: inside mid_prev/1 at offset 0xa, 1 bytes before mid_next/1; freezing and fusing mid_prev/1 and mid_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}}: inside head/1 at offset 0x1; freezing head/1{{$}}
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}}: in padding 9 bytes after the end of pad_prev/1 and 1 bytes before pad_next/1; freezing and fusing pad_prev/1 and pad_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}}: inside ret_prev/1 at offset 0x5, 1 bytes before ret_next/1; freezing and fusing ret_prev/1 and ret_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}}: inside mid_prev/1 at offset 0xa, 1 bytes before mid_next/1; freezing and fusing mid_prev/1 and mid_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}}: in padding 0 bytes after the end of end_prev/1 and 1 bytes before end_next/1; freezing and fusing end_prev/1 and end_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}}: in padding 0 bytes after the end of end_prev/1 and 1 bytes before end_next/1; freezing and fusing end_prev/1 and end_next/1
# CHECK-NOT: BOLT-ERROR

## Every reference resolves to the same address relative to both functions
## around it.
# CHECK-OUT-DAG: [[#%x,HEAD:]] t head
# CHECK-OUT-DAG: [[#%x,END_NEXT:]] t end_next
# CHECK-OUT-DAG: [[#%x,NAMED_PREV:]] t named_prev
# CHECK-OUT-DAG: [[#%x,SELF_PREV:]] t self_prev
# CHECK-OUT-DAG: [[#%x,MID_NEXT:]] t mid_next
# CHECK-OUT-DAG: [[#%x,PAD_NEXT:]] t pad_next
# CHECK-OUT-DAG: [[#%x,RET_NEXT:]] t ret_next
# CHECK-OUT: <user>:
# CHECK-OUT-NEXT: leaq {{.*}} # 0x[[#%x,PAD_NEXT-1]] <pad_prev+0xf>
# CHECK-OUT-NEXT: movl $0x[[#%x,RET_NEXT-1]], %ecx
# CHECK-OUT-NEXT: leaq {{.*}} # 0x[[#%x,MID_NEXT-1]] <mid_prev+0xa>
# CHECK-OUT-NEXT: leaq {{.*}} # 0x[[#%x,HEAD+1]] <head+0x1>
# CHECK-OUT-NEXT: movl $0x[[#%x,PAD_NEXT]], %ebx
# CHECK-OUT-NEXT: movq 0x[[#%x,PAD_NEXT-1]](%rbx), %rax
# CHECK-OUT-NEXT: movl $0x[[#%x,RET_NEXT]], %ebx
# CHECK-OUT-NEXT: movq 0x[[#%x,RET_NEXT-1]](%rbx), %rax
# CHECK-OUT-NEXT: movl $0x[[#%x,MID_NEXT]], %ebx
# CHECK-OUT-NEXT: movq 0x[[#%x,MID_NEXT-1]](%rbx), %rax

## References that are not ambiguous stay relative to the function that
## contains their target, even right before the next function: a target named
## in the symbol table, and a reference from the function itself.
# CHECK-OUT-NEXT: movl $0x[[#%x,NAMED_PREV+5]], %ecx

## From another function, one past the end of end_prev is as ambiguous as any
## other address right before end_next.
# CHECK-OUT-NEXT: leaq {{.*}} # 0x[[#%x,END_NEXT-1]] <end_prev+0x6>
# CHECK-OUT-NEXT: movl $0x[[#%x,END_NEXT-1]], %ecx

## The frozen functions keep their contents.
# CHECK-OUT: <pad_prev>:
# CHECK-OUT-NEXT: movl $0x1, %eax
# CHECK-OUT-NEXT: retq
# CHECK-OUT-COUNT-10: int3
# CHECK-OUT-EMPTY:
# CHECK-OUT-NEXT: <pad_next>:
# CHECK-OUT: <head>:
# CHECK-OUT-NEXT: nop
# CHECK-OUT-NEXT: movl $0x3, %eax
# CHECK-OUT-NEXT: retq
# CHECK-OUT: <self_prev>:
# CHECK-OUT-NEXT: movl $0x[[#%x,SELF_PREV+5]], %ecx
# CHECK-OUT-NEXT: retq

  .text
  .globl _start
  .type _start,@function
_start:
  call user
  ret
  .size _start, .-_start

## Padding between two functions.
  .section .text.pad_prev,"ax",@progbits
  .p2align 4
  .type pad_prev,@function
pad_prev:
  movl $1, %eax
  ret
  .size pad_prev, .-pad_prev

  .section .text.pad_next,"ax",@progbits
  .p2align 4
  .type pad_next,@function
pad_next:
  ret
  .size pad_next, .-pad_next

## No padding: the byte before ret_next is the ret of ret_prev.
  .section .text.ret_prev,"ax",@progbits
  .p2align 4
  .type ret_prev,@function
ret_prev:
  movl $2, %eax
  ret
  .size ret_prev, .-ret_prev

  .section .text.ret_next,"ax",@progbits
  .type ret_next,@function
ret_next:
  ret
  .size ret_next, .-ret_next

## The byte before mid_next is in the middle of an instruction of mid_prev.
  .section .text.mid_prev,"ax",@progbits
  .p2align 4
  .type mid_prev,@function
mid_prev:
  ret
  movabsq $0x1122334455667788, %rax
  .size mid_prev, .-mid_prev

  .section .text.mid_next,"ax",@progbits
  .type mid_next,@function
mid_next:
  ret
  .size mid_next, .-mid_next

## A reference one byte past the start of a function.
  .section .text.head,"ax",@progbits
  .p2align 4
  .type head,@function
head:
  nop
  movl $3, %eax
  ret
  .size head, .-head

## The byte before self_next is the ret of self_prev, which references it.
  .section .text.self_prev,"ax",@progbits
  .p2align 4
  .type self_prev,@function
self_prev:
  movl $self_next-1, %ecx
  ret
  .size self_prev, .-self_prev

  .section .text.self_next,"ax",@progbits
  .type self_next,@function
self_next:
  ret
  .size self_next, .-self_next

## The byte before named_next is the ret of named_prev, named by a local symbol
## in the symbol table.
  .section .text.named_prev,"ax",@progbits
  .p2align 4
  .type named_prev,@function
named_prev:
  movl $8, %eax
named_label:
  ret
  .size named_prev, .-named_prev

  .section .text.named_next,"ax",@progbits
  .type named_next,@function
named_next:
  ret
  .size named_next, .-named_next

## One byte of padding between end_prev and end_next.
  .section .text.end_prev,"ax",@progbits
  .p2align 4
  .type end_prev,@function
end_prev:
  movl $9, %eax
  ret
  .size end_prev, .-end_prev
  int3

  .section .text.end_next,"ax",@progbits
  .type end_next,@function
end_next:
  ret
  .size end_next, .-end_next

  .section .text.user,"ax",@progbits
  .p2align 4
  .type user,@function
user:
  leaq pad_next-1(%rip), %rax
  movl $ret_next-1, %ecx
  leaq mid_next-1(%rip), %rdx
  leaq head+1(%rip), %rsi
## The pattern of a member function pointer call after de-virtualization: a
## reference to the function, then a dead load from its vtable slot at
## "function - 1".
  movl $pad_next, %ebx
  movq pad_next-1(%rbx), %rax
  movl $ret_next, %ebx
  movq ret_next-1(%rbx), %rax
  movl $mid_next, %ebx
  movq mid_next-1(%rbx), %rax
  movl $named_label, %ecx
  leaq end_prev+6(%rip), %rax
  movl $end_prev+6, %ecx
  ret
  .size user, .-user
