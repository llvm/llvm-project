# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t

# RUN: not ld.lld %t --fix-cortex-a53-843419 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-843419
# ERR-843419: error: --fix-cortex-a53-843419 is only supported on AArch64

# RUN: not ld.lld %t --be8 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-BE8
# ERR-BE8: error: --be8 is only supported on ARM targets

# RUN: not ld.lld %t --pcrel-optimize -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-PCREL
# ERR-PCREL: error: --pcrel-optimize is only supported on PowerPC64 targets

# RUN: not ld.lld %t --toc-optimize -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR-TOC
# ERR-TOC: error: --toc-optimize is only supported on PowerPC64 targets

# RUN: not ld.lld %t -z execute-only-report=warning -o /dev/null 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ERR-EXECUTE-ONLY
# RUN: not ld.lld %t -z execute-only-report=error -o /dev/null 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ERR-EXECUTE-ONLY
# ERR-EXECUTE-ONLY: error: -z execute-only-report only supported on AArch64 and ARM

# RUN: not ld.lld %t -z execute-only-report=foo -o /dev/null 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ERR-EXECUTE-ONLY-INVALID
# ERR-EXECUTE-ONLY-INVALID: error: unknown -z execute-only-report= value: foo

.globl _start
_start:
