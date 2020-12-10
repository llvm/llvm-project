# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: not %lld %t.o -o /dev/null 2>&1 | FileCheck %s
# RUN: not %lld -demangle %t.o -o /dev/null 2>&1 | \
# RUN:     FileCheck --check-prefix=DEMANGLE %s

# CHECK: undefined symbol __Z1fv
# DEMANGLE: undefined symbol f()

.globl _main
_main:
  callq __Z1fv
  ret
