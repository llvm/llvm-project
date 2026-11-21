## Check that an indirect branch to an address loaded from a GOT entry is a
## tail call even if its block has no epilogue, as emitted with -fno-plt for
## frameless or shrink-wrapped functions. Jump tables, loads from data and GOT
## entries pointing inside the function at a non-entry offset stay rejected.
## With BTI, functions with such a tail call are still not inlined.

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/lib.s \
# RUN:   -o %t/lib.o
# RUN: %clang %cflags %t/lib.o -o %t/lib.so -shared -Wl,-q -Wl,--no-relax
# RUN: llvm-bolt %t/lib.so -o %t/lib.bolt.so --print-cfg \
# RUN:   --print-only=frameless,pre_prologue -v=2 > %t/lib.log 2>&1
# RUN: FileCheck %s --check-prefix=FRAMELESS --input-file=%t/lib.log
# RUN: FileCheck %s --check-prefix=PRE-PROLOGUE --input-file=%t/lib.log
# RUN: FileCheck %s --check-prefix=REJECTED --input-file=%t/lib.log

## Without dynamic relocations, the GOT entry holds the target address.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/static.s \
# RUN:   -o %t/static.o
# RUN: %clang %cflags %t/static.o -o %t/static.exe -static -Wl,-q \
# RUN:   -Wl,--no-relax
# RUN: llvm-bolt %t/static.exe -o %t/static.bolt --print-cfg \
# RUN:   --print-only=external -v=2 2>&1 | FileCheck %s --check-prefix=STATIC

## Point the only GOT entry at a block of the same function that is not an
## entry point. A GOT relocation against a local label would assert in the
## symbolizer, so patch the entry after linking.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/internal.s \
# RUN:   -o %t/internal.o
# RUN: %clang %cflags %t/internal.o -o %t/internal.exe -static -Wl,-q \
# RUN:   -Wl,--no-relax -Wl,--image-base=0,--section-start=.text=0x10000
# RUN: llvm-objdump -d --no-show-raw-insn %t/internal.exe \
# RUN:   | FileCheck %s --check-prefix=LAYOUT
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/got.s \
# RUN:   -o %t/got.o
# RUN: llvm-objcopy -O binary --only-section=.data %t/got.o %t/got.bin
# RUN: llvm-objcopy --update-section=.got=%t/got.bin %t/internal.exe \
# RUN:   %t/internal.patched.exe
# RUN: llvm-bolt %t/internal.patched.exe -o %t/internal.bolt -v=2 2>&1 \
# RUN:   | FileCheck %s --check-prefix=INTERNAL

## Inlining would turn the br into a blr, which needs a different landing pad
## at the unknown target.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/bti.s \
# RUN:   -o %t/bti.o
# RUN: %clang %cflags %t/bti.o -o %t/bti.exe -static -Wl,-q -Wl,--no-relax \
# RUN:   -Wl,-z,force-bti
# RUN: llvm-bolt %t/bti.exe -o %t/bti.bolt --inline-all --print-cfg \
# RUN:   --print-finalized --print-only=_start,bti_tail -v=2 2>&1 \
# RUN:   | FileCheck %s --check-prefix=BTI

# FRAMELESS:      Binary Function "frameless" after building cfg
# FRAMELESS:        IsSimple    : 1
# FRAMELESS:        br x16 # TAILCALL
# FRAMELESS:      End of Function "frameless"

# PRE-PROLOGUE:   Binary Function "pre_prologue" after building cfg
# PRE-PROLOGUE:     IsSimple    : 1
# PRE-PROLOGUE:     br x1 # TAILCALL
# PRE-PROLOGUE:   End of Function "pre_prologue"

# REJECTED-DAG:   rejected potential indirect tail call in function jump_table
# REJECTED-DAG:   rejected potential indirect tail call in function data_ptr
# REJECTED-NOT:   rejected potential indirect tail call in function frameless
# REJECTED-NOT:   rejected potential indirect tail call in function pre_prologue

# STATIC-NOT:     rejected potential indirect tail call
# STATIC:         Binary Function "external" after building cfg
# STATIC:           IsSimple    : 1
# STATIC:           br x16 # TAILCALL

# LAYOUT:         0000000000010000 <internal>:
# LAYOUT:         10010: ret

# INTERNAL:       rejected potential indirect tail call in function internal

# BTI:            BOLT-INFO: binary is using BTI
# BTI-NOT:        rejected potential indirect tail call
# BTI:            Binary Function "bti_tail" after building cfg
# BTI:              IsSimple    : 1
# BTI:              br x16 # TAILCALL
# BTI:            Binary Function "_start" after finalize-functions
# BTI:              bl bti_tail

#--- lib.s
  .text
  .globl frameless
  .type frameless, %function
frameless:
  cbz x0, .Lframeless_ret
  adrp x16, :got:ext
  ldr x16, [x16, :got_lo12:ext]
  br x16
.Lframeless_ret:
  ret
  .size frameless, .-frameless

## The GOT tail call runs before the prologue of the slow path.
  .globl pre_prologue
  .type pre_prologue, %function
pre_prologue:
  cbnz x0, .Lpre_prologue_slow
  adrp x1, :got:callee
  ldr x1, [x1, :got_lo12:callee]
  br x1
.Lpre_prologue_slow:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl callee
  ldp x29, x30, [sp], #16
  ret
  .size pre_prologue, .-pre_prologue

## The table address comes from a GOT entry, the branch target does not.
  .globl jump_table
  .type jump_table, %function
jump_table:
  adrp x3, :got:jt
  ldr x3, [x3, :got_lo12:jt]
  ldrh w3, [x3, x1, lsl #1]
  adr x1, .Ljt_0
  add x3, x1, w3, sxth #2
  br x3
.Ljt_0:
  ret
.Ljt_1:
  ret
  .size jump_table, .-jump_table

  .globl data_ptr
  .type data_ptr, %function
data_ptr:
  cbz x0, .Ldata_ptr_ret
  adrp x16, fptr
  ldr x16, [x16, :lo12:fptr]
  br x16
.Ldata_ptr_ret:
  ret
  .size data_ptr, .-data_ptr

  .globl callee
  .type callee, %function
callee:
  ret
  .size callee, .-callee

## Preemptible, so references to it still go through the GOT.
  .globl ext
  .type ext, %function
ext:
  ret
  .size ext, .-ext

  .section .rodata,"a",@progbits
jt:
  .hword (.Ljt_0-.Ljt_0)>>2
  .hword (.Ljt_1-.Ljt_0)>>2

  .data
  .p2align 3
fptr:
  .xword callee

#--- static.s
  .text
  .globl _start
  .type _start, %function
_start:
  bl external
  ret
  .size _start, .-_start

  .globl external
  .type external, %function
external:
  cbz x0, .Lexternal_ret
  adrp x16, :got:callee
  ldr x16, [x16, :got_lo12:callee]
  br x16
.Lexternal_ret:
  ret
  .size external, .-external

  .globl callee
  .type callee, %function
callee:
  ret
  .size callee, .-callee

#--- internal.s
  .text
  .globl internal
  .type internal, %function
internal:
  cbz x0, .Linner
  adrp x16, :got:callee
  ldr x16, [x16, :got_lo12:callee]
  br x16
.Linner:
  ret
  .size internal, .-internal

  .globl _start
  .type _start, %function
_start:
  bl internal
  ret
  .size _start, .-_start

  .globl callee
  .type callee, %function
callee:
  ret
  .size callee, .-callee

#--- got.s
## Address of .Linner in internal.exe.
  .data
  .xword 0x10010

#--- bti.s
  .text
  .globl _start
  .type _start, %function
_start:
  bti c
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  bl bti_tail
  ldp x29, x30, [sp], #16
  ret
  .size _start, .-_start

  .globl bti_tail
  .type bti_tail, %function
bti_tail:
  bti c
  cbz x0, .Lbti_tail_ret
  adrp x16, :got:callee
  ldr x16, [x16, :got_lo12:callee]
  br x16
.Lbti_tail_ret:
  ret
  .size bti_tail, .-bti_tail

  .globl callee
  .type callee, %function
callee:
  bti c
  ret
  .size callee, .-callee
