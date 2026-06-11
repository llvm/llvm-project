## This test verifies the effect of icp on aarch64 and inline after icp.

## The assembly was produced from C code compiled with clang -O1 -S:

# int foo(int x) { return x + 1; }
# int bar(int x) { return x*100 + 42; }
# typedef int (*const fn)(int);
# fn funcs[] = { foo, bar };
#
# int indirectTailCall(int argc) {
#   fn func = funcs[argc];
#   return func(0);
# }
# int indirectCall(int argc) {
#   fn func = funcs[argc];
#   int result = func(0);
#   __asm__ volatile("" ::: "memory");
#   return result;
# }

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib -pie

# AArch64 jump table ICP is disabled until BOLT can reliably recover the
# jump table index register and limits.
# RUN: not llvm-bolt %t.exe --icp=jump-tables -o %t.null --lite=0 --assume-abi \
# RUN:   --data %t.fdata \
# RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-ICP-JT-DISABLED
# CHECK-ICP-JT-DISABLED: BOLT-ERROR: ICP jump table promotion is disabled on AArch64

# Indirect call promotion without inline
# RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 \
# RUN:   -o %t.null --lite=0 --assume-abi --print-icp --data %t.fdata \
# RUN:   | FileCheck %s --check-prefix=CHECK-ICP-NO-INLINE
# CHECK-ICP-NO-INLINE: Binary Function "indirectTailCall" after indirect-call-promotion
# CHECK-ICP-NO-INLINE: b    bar
# CHECK-ICP-NO-INLINE: End of Function "indirectTailCall"
# CHECK-ICP-NO-INLINE: Binary Function "indirectCall" after indirect-call-promotion
# CHECK-ICP-NO-INLINE: bl    bar
# CHECK-ICP-NO-INLINE: End of Function "indirectCall"

# Indirect call promotion with inline
# RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 --inline-small-functions \
# RUN:   -o %t.null --lite=0 --assume-abi --inline-small-functions-bytes=12 \
# RUN:   --print-inline --data %t.fdata \
# RUN:   | FileCheck %s --check-prefix=CHECK-ICP-WITH-INLINE
# CHECK-ICP-WITH-INLINE:     Binary Function "indirectTailCall" after inlining
# CHECK-ICP-WITH-INLINE:     br    x1
# CHECK-ICP-WITH-INLINE-NOT: b    bar
# CHECK-ICP-WITH-INLINE:     End of Function "indirectTailCall"
    .globl  foo
    .type   foo,@function
foo:
    .cfi_startproc
    add     w0, w0, #1
    ret
    .Lfunc_end0:
    .size   foo, .Lfunc_end0-foo
    .cfi_endproc

    .globl  bar
    .type   bar,@function
bar:
    .cfi_startproc
    mov     w8, #100
    mov     x9, #42
    madd    w0, w0, w8, w9
    ret
.Lfunc_end1:
    .size   bar, .Lfunc_end1-bar
    .cfi_endproc

    .globl  indirectTailCall
    .type   indirectTailCall,@function
indirectTailCall:
    .cfi_startproc
    adrp    x8, funcs
    add     x8, x8, :lo12:funcs
    ldr     x1, [x8, w0, sxtw #3]
    mov     w0, wzr
    br     x1
# FDATA: 1 indirectTailCall 10 1 foo 0 0 1
# FDATA: 1 indirectTailCall 10 1 bar 0 0 2
.Lfunc_end2:
    .size   indirectTailCall, .Lfunc_end2-indirectTailCall
    .cfi_endproc

    .globl  indirectCall
    .type   indirectCall,@function
indirectCall:
    .cfi_startproc
    stp     x29, x30, [sp, #-16]!
    .cfi_def_cfa_offset 16
    .cfi_offset w29, -16
    .cfi_offset w30, -8
    mov     x29, sp
    adrp    x8, funcs
    add     x8, x8, :lo12:funcs
    ldr     x1, [x8, w0, sxtw #3]
    mov     w0, wzr
    blr     x1
# FDATA: 1 indirectCall 18 1 foo 0 0 1
# FDATA: 1 indirectCall 18 1 bar 0 0 2
    ldp     x29, x30, [sp], #16
    ret
.Lfunc_end3:
    .size   indirectCall, .Lfunc_end3-indirectCall
    .cfi_endproc

    .type   funcs,@object
    .section    .data.rel.ro,"aw",@progbits
    .globl  funcs
funcs:
    .xword  foo
    .xword  bar
    .size   funcs, 16
