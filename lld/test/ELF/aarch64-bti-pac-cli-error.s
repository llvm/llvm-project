# REQUIRES: x86
# RUN: llvm-mc --triple=x86_64-pc-linux --filetype=obj -o %t.o %s
# RUN: not ld.lld -z pac-plt -z force-bti -z bti-report=error %t.o -o /dev/null 2>&1 | FileCheck %s
#
## Check that we error if -z pac-plt, -z force-bti and -z bti-report=error are used when target is not
## aarch64

# CHECK: error: -z pac-plt only supported on AArch64
# CHECK-NEXT: error: -z force-bti only supported on AArch64
# CHECK-NEXT: error: -z bti-report only supported on AArch64

# RUN: not ld.lld -z bti-report=something %t.o -o /dev/null 2>&1 | \
# RUN:     FileCheck --check-prefix=REPORT_INVALID %s
# REPORT_INVALID: error: -z bti-report= parameter something is not recognized
# REPORT_INVALID-EMPTY:

        .globl start
start:  ret
