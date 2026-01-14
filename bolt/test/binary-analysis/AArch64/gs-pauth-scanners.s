// RUN: %clang %cflags -march=armv8.3-a %s -o %t.exe

// Select single detector:
//
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-pac-ret      %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-tail-calls   %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=TAIL-CALLS-COMMON,TAIL-CALLS-NOFPAC
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-forward-cf   %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=FORWARD-CF
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-sign-oracles %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=SIGN-ORACLES-COMMON,SIGN-ORACLES-NOFPAC
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-auth-oracles %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=AUTH-ORACLES-NOFPAC

// Select multiple options (either disjoint or not):
//
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-pac-ret,ptrauth-forward-cf %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET,FORWARD-CF
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-pac-ret,ptrauth-all %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET,TAIL-CALLS-COMMON,TAIL-CALLS-NOFPAC,FORWARD-CF,SIGN-ORACLES-COMMON,SIGN-ORACLES-NOFPAC,AUTH-ORACLES-NOFPAC

// Select one of "all" options:
//
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-all %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET,TAIL-CALLS-COMMON,TAIL-CALLS-NOFPAC,FORWARD-CF,SIGN-ORACLES-COMMON,SIGN-ORACLES-NOFPAC,AUTH-ORACLES-NOFPAC
// RUN: llvm-bolt-binary-analysis --scanners=all %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET,TAIL-CALLS-COMMON,TAIL-CALLS-NOFPAC,FORWARD-CF,SIGN-ORACLES-COMMON,SIGN-ORACLES-NOFPAC,AUTH-ORACLES-NOFPAC

// Test FPAC handling:
//
// RUN: llvm-bolt-binary-analysis --auth-traps-on-failure --scanners=ptrauth-all %t.exe 2>&1 | \
// RUN:     FileCheck %s --implicit-check-not="found in function" \
// RUN:                  --check-prefixes=PACRET,TAIL-CALLS-COMMON,FORWARD-CF,SIGN-ORACLES-COMMON
// RUN: llvm-bolt-binary-analysis --auth-traps-on-failure --scanners=ptrauth-auth-oracles %t.exe 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=NO-REPORTS

// NO-REPORTS-NOT: found in function

        .text

        .globl  callee
        .type   callee,@function
callee:
        ret
        .size callee, .-callee

        .globl  bad_pacret
        .type   bad_pacret,@function
bad_pacret:
// PACRET: GS-PAUTH: non-protected ret found in function bad_pacret
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldp     x29, x30, [sp], #16
        ret
        .size bad_pacret, .-bad_pacret

        .globl  bad_tail_call_common
        .type   bad_tail_call_common,@function
bad_tail_call_common:
// TAIL-CALLS-COMMON: GS-PAUTH: untrusted link register found before tail call in function bad_tail_call_common
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldp     x29, x30, [sp], #16
        b       callee
        .size bad_tail_call_common, .-bad_tail_call_common

        .globl  bad_tail_call_nofpac
        .type   bad_tail_call_nofpac,@function
bad_tail_call_nofpac:
// TAIL-CALLS-NOFPAC:   GS-PAUTH: untrusted link register found before tail call in function bad_tail_call_nofpac
// AUTH-ORACLES-NOFPAC: GS-PAUTH: authentication oracle found in function bad_tail_call_nofpac
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldp     x29, x30, [sp], #16
        autiasp
        b       callee
        .size bad_tail_call_nofpac, .-bad_tail_call_nofpac

        .globl  bad_call
        .type   bad_call,@function
bad_call:
// FORWARD-CF: GS-PAUTH: non-protected call found in function bad_call
        br      x0
        .size bad_call, .-bad_call

        .globl  bad_signing_oracle_common
        .type   bad_signing_oracle_common,@function
bad_signing_oracle_common:
// SIGN-ORACLES-COMMON: GS-PAUTH: signing oracle found in function bad_signing_oracle_common
        pacda   x0, x1
        ret
        .size bad_signing_oracle_common, .-bad_signing_oracle_common

        .globl  bad_signing_oracle_nofpac
        .type   bad_signing_oracle_nofpac,@function
bad_signing_oracle_nofpac:
// SIGN-ORACLES-NOFPAC: GS-PAUTH: signing oracle found in function bad_signing_oracle_nofpac
// AUTH-ORACLES-NOFPAC: GS-PAUTH: authentication oracle found in function bad_signing_oracle_nofpac
        autda   x0, x1
        pacdb   x0, x1
        ret
        .size bad_signing_oracle_nofpac, .-bad_signing_oracle_nofpac

        .globl  bad_auth_oracle
        .type   bad_auth_oracle,@function
bad_auth_oracle:
// AUTH-ORACLES-NOFPAC: GS-PAUTH: authentication oracle found in function bad_auth_oracle
        autda   x0, x1
        ret
        .size bad_auth_oracle, .-bad_auth_oracle
