# Test that BOLT will produce error by default and pass with instrumentation-sleep-time option

# REQUIRES: system-linux,bolt-runtime,target=x86_64-{{.*}}

# RUN: llvm-mc -triple x86_64 -filetype=obj %s -o %t.o
# RUN: ld.lld -q -pie -o %t.exe %t.o
# RUN: llvm-readelf -d %t.exe | FileCheck --check-prefix=CHECK-NO-FINI %s
# RUN: not llvm-bolt --instrument -o %t.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-BOLT-FAIL
# RUN: llvm-bolt --instrument --instrumentation-sleep-time=1 -o %t.out %t.exe 2>&1 | FileCheck %s --check-prefix=CHECK-BOLT-PASS

# CHECK-NO-FINI: INIT
# CHECK-NO-FINI-NOT: FINI
# CHECK-NO-FINI-NOT: FINI_ARRAY

# CHECK-BOLT-FAIL: Instrumentation needs either DT_FINI or DT_FINI_ARRAY

# CHECK-BOLT-PASS-NOT: Instrumentation needs either DT_FINI or DT_FINI_ARRAY
# CHECK-BOLT-PASS: runtime library initialization was hooked via DT_INIT

    .text
    .globl _start
    .type _start, %function
_start:
    # BOLT errs when instrumenting without relocations; create a dummy one.
    .reloc 0, R_X86_64_NONE
    retq
    .size _start, .-_start

    .globl _init
    .type _init, %function
    # Force DT_INIT to be created (needed for instrumentation).
_init:
    retq
    .size _init, .-_init
