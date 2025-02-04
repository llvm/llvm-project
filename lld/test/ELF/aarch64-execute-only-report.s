// REQUIRES: aarch64

// RUN: llvm-mc --triple=aarch64-linux-none --filetype=obj -o %t.o %s
// RUN: ld.lld -z execute-only-report=none --fatal-warnings %t.o -o /dev/null 2>&1
// RUN: ld.lld -z execute-only-report=warning %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=WARNING %s
// RUN: not ld.lld -z execute-only-report=error %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=ERROR %s

// WARNING: warning: {{.*}}.o: -z execute-only-report: section .text does not have SHF_AARCH64_PURECODE flag set
// ERROR: error: {{.*}}.o: -z execute-only-report: section .text does not have SHF_AARCH64_PURECODE flag set

        .section .text,"ax"
        .globl _start
_start:
        ret
