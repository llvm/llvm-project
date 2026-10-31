## This checks that shrink wrapping places a restore for a callee-saved
## register before a conditional tail call.  A conditional tail call (e.g.
## introduced by -indirect-call-promotion) exits the function via an edge to an
## external symbol that is not represented as a CFG successor, so it is absent
## from the dominance frontier used for restore placement.  Without the fix, the
## promoted tail-call path skipped the restore and leaked the clobbered
## callee-saved register to the caller.
##
## See https://github.com/llvm/llvm-project/issues/236353.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe --relocs -o %t.out --data %t.fdata \
# RUN:   --frame-opt=all --eliminate-unreachable=false 2>&1 | FileCheck %s
# RUN: llvm-objdump -d %t.out --print-imm-hex | \
# RUN:   FileCheck --check-prefix=CHECK-OBJDUMP %s

# CHECK: BOLT-INFO: Shrink wrapping moved 1 spills inserting load/stores and 0 spills inserting push/pops

# The restore (movq (%rsp), %rbx) must be emitted before the conditional tail
# call (je <ext_func>).  Without the fix, the conditional tail call comes first
# and its taken edge skips the restore.
# CHECK-OBJDUMP:      movq (%rsp), %rbx
# CHECK-OBJDUMP-NEXT: je {{.*}} <ext_func>

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  push  %rbx
  cmpq  $0, %rdi
br0:
  je    cold_path
# FDATA: 1 _start #br0# 1 _start #cold_path# 0 90
use:
  mov   %rdi, %rbx
  cmpq  $0, %rsi
br1:
  je    ctc_path
# FDATA: 1 _start #br1# 1 _start #ctc_path# 0 3
use_cont:
  mov   %rbx, %rax
  pop   %rbx
  ret
# ctc_path models the conditional tail call left by -indirect-call-promotion:
# the taken edge of `je ext_func` is a tail call that exits the function, so
# %rbx must be restored there too.  The fall-through `pop %rbx` only covers the
# non-tail-call edge.
ctc_path:
  je    ext_func
  pop   %rbx
  ret
cold_path:
  pop   %rbx
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl ext_func
  .type ext_func, %function
ext_func:
  ret
  .size ext_func, .-ext_func
