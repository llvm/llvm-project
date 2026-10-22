## Verify that binary analyses warn about functions for which BOLT could not
## reconstruct the CFG, since analysis results are less reliable for them.

// RUN: %clang %cflags %s %p/../../Inputs/asm_main.c -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-pac-ret %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=SUMMARY %s
// RUN: llvm-bolt-binary-analysis --scanners=ptrauth-pac-ret -v=1 %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=VERBOSE %s

        .text

## A function with a regular CFG must not be reported.
        .globl  f_good
        .type   f_good,@function
f_good:
        ret
        .size f_good, .-f_good
// SUMMARY-NOT: BOLT-WARNING:{{.*}}f_good
// VERBOSE-NOT: BOLT-WARNING:{{.*}}f_good

## An unanalyzable indirect branch prevents BOLT from building the CFG.
        .globl  f_nocfg
        .type   f_nocfg,@function
f_nocfg:
        adr     x2, 1f
        br      x2
1:
        ret
        .size f_nocfg, .-f_nocfg

## Without -v, only the aggregate warning is emitted; functions are not listed
## individually.
// SUMMARY-NOT: BOLT-WARNING: no CFG for
// SUMMARY:     BOLT-WARNING: {{[0-9]+}} function(s) lack CFG; binary-analysis results may be incomplete. Re-run with -v=1 to list these functions.

## With -v=1, each function lacking a CFG is listed before the summary.
// VERBOSE:     BOLT-WARNING: no CFG for {{.*}}f_nocfg{{.*}}; binary analyses may be imprecise
// VERBOSE:     BOLT-WARNING: {{[0-9]+}} function(s) lack CFG; binary-analysis results may be incomplete. Re-run with -v=1 to list these functions.
