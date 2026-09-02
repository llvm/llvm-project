# Test --instrument-funcs-file, alone and combined with --instrument-hot-only.
#
# The binary defines three functions (foo, bar, baz). We attach a profile that
# only marks foo as hot. With --instrument-funcs-file listing foo and bar, only
# those two are instrumented. Adding --instrument-hot-only further restricts
# instrumentation to foo (the only function that is both listed and hot).

# REQUIRES: system-linux,bolt-runtime,target=x86_64-{{.*}}

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: link_fdata %s %t.exe %t.fdata

# Funcs file lists foo and bar (baz is intentionally omitted).
# RUN: echo "foo" > %t.funcs
# RUN: echo "bar" >> %t.funcs

# Test A: only --instrument-funcs-file. Both foo and bar get instrumented.
# RUN: llvm-bolt --instrument --instrument-funcs-file=%t.funcs \
# RUN:     -o %t.a.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-A

# Test B: --instrument-funcs-file combined with --instrument-hot-only. Profile
# marks only foo as hot, so bar is filtered out by --instrument-hot-only.
# RUN: llvm-bolt --instrument --instrument-funcs-file=%t.funcs \
# RUN:     --instrument-hot-only --data %t.fdata \
# RUN:     -o %t.b.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-B

# Test C: empty file means "no functions match", so nothing is instrumented.
# RUN: rm -f %t.empty && touch %t.empty
# RUN: llvm-bolt --instrument --instrument-funcs-file=%t.empty \
# RUN:     -o %t.c.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-C

# Test D: missing file produces a fatal error.
# RUN: not llvm-bolt --instrument --instrument-funcs-file=%t.missing \
# RUN:     -o %t.d.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-D

# CHECK-A: BOLT-INSTRUMENTER: Number of function descriptors: 2
# CHECK-B: BOLT-INSTRUMENTER: Number of function descriptors: 1
# CHECK-C: BOLT-INSTRUMENTER: Number of function descriptors: 0
# CHECK-D: instrument-funcs-file {{.*}}.missing{{.*}} can't be opened

    .text
    .globl _start
    .type _start, %function
_start:
    call foo
    call bar
    call baz
    retq
    .size _start, .-_start

    .globl foo
    .type foo, %function
foo:
# FDATA: 0 [unknown] 0 1 foo 0 0 100
    retq
    .size foo, .-foo

    .globl bar
    .type bar, %function
bar:
    retq
    .size bar, .-bar

    .globl baz
    .type baz, %function
baz:
    retq
    .size baz, .-baz

    .globl _init
    .type _init, %function
    # Force DT_INIT to be created (needed for instrumentation).
_init:
    retq
    .size _init, .-_init

    .globl _fini
    .type _fini, %function
    # Force DT_FINI to be created (needed for instrumentation).
_fini:
    retq
    .size _fini, .-_fini
