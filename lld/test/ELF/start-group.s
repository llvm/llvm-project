# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: not ld.lld --start-group --start-group %t.o 2>&1 | FileCheck %s --check-prefix=NESTED
# NESTED: nested --start-group

# RUN: not ld.lld --end-group 2>&1 | FileCheck %s --check-prefix=END
# RUN: not ld.lld '-)' 2>&1 | FileCheck %s --check-prefix=END
# END: stray --end-group

.globl _start
_start:
