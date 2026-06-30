## This test verifies basic indirect call promotion on AArch64.

## The assembly models the following C code:

# void hello(void) {}
#
# void execute(void (*callback)(void)) {
#   callback();
# }
#
# int exit_success(void) {
#   return 0;
# }
#
# int executeTailCall(int (*callback)(void)) {
#   return callback();
# }
#
# int main(void) {
#   execute(hello);
#   return executeTailCall(exit_success);
# }

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib -pie

# RUN: llvm-bolt %t.exe --icp=calls --icp-calls-topn=1 \
# RUN:   -o %t.null --lite=0 --assume-abi --print-icp --data %t.fdata \
# RUN:   | FileCheck %s
# CHECK: Binary Function "execute" after indirect-call-promotion
# CHECK: bl    hello
# CHECK: End of Function "execute"
# CHECK: Binary Function "executeTailCall" after indirect-call-promotion
# CHECK: b    exit_success
# CHECK: End of Function "executeTailCall"

    .text
    .globl  hello
    .type   hello,@function
hello:
    .cfi_startproc
    ret
.Lfunc_end0:
    .size   hello, .Lfunc_end0-hello
    .cfi_endproc

    .globl  execute
    .type   execute,@function
execute:
    .cfi_startproc
    stp     x29, x30, [sp, #-16]!
    .cfi_def_cfa_offset 16
    .cfi_offset w29, -16
    .cfi_offset w30, -8
    mov     x29, sp
execute_call:
    blr     x0
# FDATA: 1 execute #execute_call# 1 hello 0 0 1
    ldp     x29, x30, [sp], #16
    ret
.Lfunc_end1:
    .size   execute, .Lfunc_end1-execute
    .cfi_endproc

    .globl  exit_success
    .type   exit_success,@function
exit_success:
    .cfi_startproc
    mov     w0, wzr
    ret
.Lfunc_end2:
    .size   exit_success, .Lfunc_end2-exit_success
    .cfi_endproc

    .globl  executeTailCall
    .type   executeTailCall,@function
executeTailCall:
    .cfi_startproc
execute_tail_call:
    br      x0
# FDATA: 1 executeTailCall #execute_tail_call# 1 exit_success 0 0 1
.Lfunc_end3:
    .size   executeTailCall, .Lfunc_end3-executeTailCall
    .cfi_endproc

    .globl  main
    .type   main,@function
main:
    .cfi_startproc
    stp     x29, x30, [sp, #-16]!
    .cfi_def_cfa_offset 16
    .cfi_offset w29, -16
    .cfi_offset w30, -8
    mov     x29, sp
    adrp    x0, hello
    add     x0, x0, :lo12:hello
    bl      execute
    adrp    x0, exit_success
    add     x0, x0, :lo12:exit_success
    bl      executeTailCall
    ldp     x29, x30, [sp], #16
    ret
.Lfunc_end4:
    .size   main, .Lfunc_end4-main
    .cfi_endproc
