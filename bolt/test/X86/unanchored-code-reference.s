## Check that BOLT reports non-branch references into code that are not
## anchored on a symbol and land in padding or near a function start, and that,
## in lite mode, it keeps the functions such a reference depends on at their
## original addresses.
##
## Such references are ambiguous. HHVM built with LLVM 23 has one in sqlite:
##
##   lea  -0x51894(%rip),%rdx      # sqlite3RCStrUnref - 1
##   cmp  $0xfffffffffffffffd,%rdx
##
## "xDel == 0 || xDel == (void *)-1" became "(uintptr_t)xDel - 1 >u -3" before
## xDel was folded to sqlite3RCStrUnref, so the value is only compared. The
## function is static: the assembler emits the relocation against its section
## symbol, ".text.sqlite3RCStrUnref - 5", and "ld --emit-relocs" rewrites it to
## ".text + offset". In the input, "sqlite3RCStrUnref - 1" is also the last byte
## of the padding after the preceding function, and nothing tells which of the
## two the code means. They differ as soon as either function moves.
##
## Without padding, "Next - 1" is the last byte of the preceding function: a ret,
## which BOLT used to turn silently into a secondary entry point of that
## function, or the middle of an instruction. The same "f - 1" comes from member
## function pointer calls after de-virtualization: the dead branch for virtual
## functions loads from "vtable + ptr - 1", which becomes "movq f-1(%reg)".
##
## References from code are checked: RIP-relative operands without a relocation
## and absolute relocations against section symbols. References from data,
## from the function that contains the target, to a target named in the symbol
## table, or farther than --boundary-ref-distance bytes are not acted on.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --nostdlib -e _start
# RUN: echo user > %t.order
# RUN: echo pad_prev >> %t.order
# RUN: echo pad_next >> %t.order
# RUN: echo ret_prev >> %t.order
# RUN: echo ret_next >> %t.order
# RUN: echo head >> %t.order
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs --lite --reorder-functions=user \
# RUN:   --function-order=%t.order --skip-funcs=cold_user/1 2>&1 | FileCheck %s
# RUN: llvm-nm %t.exe %t.bolt | FileCheck %s --check-prefix=CHECK-NM

# CHECK: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}} from function user/1 at {{.*}}: in padding 9 bytes after the end of pad_prev/1 and 1 bytes before pad_next/1; ignoring pad_prev/1 and pad_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored absolute reference to {{.*}} from function user/1 at {{.*}}: inside ret_prev/1 at offset 0x5, 1 bytes before ret_next/1; ignoring ret_prev/1 and ret_next/1
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}} from function user/1 at {{.*}}: inside head/1 at offset 0x1; ignoring head/1{{$}}

## Functions that are not disassembled, like cold_user, are scanned too.
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}} from function cold_user/1 at {{.*}}: in padding 9 bytes after the end of cold_prev/1 and 1 bytes before cold_next/1; ignoring cold_prev/1 and cold_next/1

## The contents of the gap do not matter: the last byte of a constant table is
## as ambiguous as padding. A reference farther from the next function, like
## the start of the table, is not reported.
# CHECK-NEXT: BOLT-WARNING: unanchored PC-relative LEA reference to {{.*}} from function table_user/1 at {{.*}}: in padding 25 bytes after the end of table_prev/1 and 1 bytes before table_next/1; ignoring table_prev/1 and table_next/1

## References from data, like the entries of a jump table that point right
## before the next function, are not reported.

## The reference anchored on a global symbol, the one to a target named in the
## symbol table, and the one to the start of pad_prev are not counted. The one past the end of pad_prev is 10
## bytes before pad_next, i.e. between functions.
# CHECK-NEXT: BOLT-INFO: unanchored non-branch references into code within 2 bytes of a function start: 3 in padding before it, 1 inside the preceding function, 1 after it; elsewhere: 1 to functions of at most 2 bytes, 2 between functions, 0 inside functions

## The ignored functions stay at their original addresses even though the
## function order lists them.
# CHECK-NM: .exe:
# CHECK-NM: [[HEAD:[0-9a-f]+]] t head
# CHECK-NM: [[PAD_NEXT:[0-9a-f]+]] t pad_next
# CHECK-NM: [[PAD_PREV:[0-9a-f]+]] t pad_prev
# CHECK-NM: [[RET_NEXT:[0-9a-f]+]] t ret_next
# CHECK-NM: [[RET_PREV:[0-9a-f]+]] t ret_prev
# CHECK-NM: .bolt:
# CHECK-NM: [[HEAD]] t head
# CHECK-NM: [[PAD_NEXT]] t pad_next
# CHECK-NM: [[PAD_PREV]] t pad_prev
# CHECK-NM: [[RET_NEXT]] t ret_next
# CHECK-NM: [[RET_PREV]] t ret_prev

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

## The start of a function that is only one byte long.
  .section .text.tiny,"ax",@progbits
  .p2align 4
  .type tiny,@function
tiny:
  ret
  .size tiny, .-tiny

  .section .text.after_tiny,"ax",@progbits
  .type after_tiny,@function
after_tiny:
  ret
  .size after_tiny, .-after_tiny

## A reference one byte past the start of a function.
  .section .text.head,"ax",@progbits
  .p2align 4
  .type head,@function
head:
  movl $3, %eax
  ret
  .size head, .-head

## A target named by a local symbol in the symbol table, right before the next
## function. The reference to it is against the section symbol.
  .section .text.named_prev,"ax",@progbits
  .p2align 4
  .type named_prev,@function
named_prev:
  movl $7, %eax
named_label:
  ret
  .size named_prev, .-named_prev

  .section .text.named_next,"ax",@progbits
  .type named_next,@function
named_next:
  ret
  .size named_next, .-named_next

## A global function: references to it are anchored on its symbol.
  .section .text.anchored,"ax",@progbits
  .p2align 4
  .globl anchored
  .type anchored,@function
anchored:
  ret
  .size anchored, .-anchored

  .section .text.user,"ax",@progbits
  .p2align 4
  .type user,@function
user:
  leaq pad_next-1(%rip), %rax
  movl $ret_next-1, %ecx
  leaq after_tiny-1(%rip), %rdx
  leaq pad_prev+6(%rip), %rsi
  leaq anchored-1(%rip), %rdi
  leaq named_label(%rip), %r8
  leaq head+1(%rip), %r9
  leaq pad_prev(%rip), %r10
  ret
  .size user, .-user

## A function with an absolute jump table whose entries point at its last
## instructions, right before the next function.
  .section .text.switch,"ax",@progbits
  .p2align 4
  .type switch,@function
switch:
  jmpq *table(,%rdi,8)
1:
  movl $4, %eax
2:
  ret
  .size switch, .-switch

  .section .text.after_switch,"ax",@progbits
  .type after_switch,@function
after_switch:
  ret
  .size after_switch, .-after_switch

  .section .text.cold_prev,"ax",@progbits
  .p2align 4
  .type cold_prev,@function
cold_prev:
  movl $5, %eax
  ret
  .size cold_prev, .-cold_prev

  .section .text.cold_next,"ax",@progbits
  .p2align 4
  .type cold_next,@function
cold_next:
  ret
  .size cold_next, .-cold_next

## Skipped, so it is scanned instead of disassembled.
  .section .text.cold_user,"ax",@progbits
  .p2align 4
  .type cold_user,@function
cold_user:
  leaq cold_next-1(%rip), %rax
  ret
  .size cold_user, .-cold_user

## A constant table between two functions, as in hand-written assembly.
  .section .text.table_prev,"ax",@progbits
  .p2align 4
  .type table_prev,@function
table_prev:
  movl $6, %eax
  ret
  .size table_prev, .-table_prev
  .p2align 4
.Lconstants:
  .quad 0x1122334455667788, 0x99aabbccddeeff00

  .section .text.table_next,"ax",@progbits
  .type table_next,@function
table_next:
  ret
  .size table_next, .-table_next

  .section .text.table_user,"ax",@progbits
  .p2align 4
  .type table_user,@function
table_user:
  leaq .Lconstants(%rip), %rax
  leaq table_next-1(%rip), %rdx
  ret
  .size table_user, .-table_user

  .section .rodata,"a",@progbits
table:
  .quad 1b
  .quad 2b
