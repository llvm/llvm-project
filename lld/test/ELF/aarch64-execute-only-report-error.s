// REQUIRES: x86

// RUN: llvm-mc --triple=x86_64-pc-linux --filetype=obj -o %t.o %s
// RUN: not ld.lld -z execute-only-report=warning %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck %s
// RUN: not ld.lld -z execute-only-report=error %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck %s

// CHECK: error: -z execute-only-report only supported on AArch64
//
// RUN: not ld.lld -z execute-only-report=something %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=REPORT_INVALID %s
// REPORT_INVALID: error: -z execute-only-report= parameter something is not recognized

        .globl _start
_start:
        ret
