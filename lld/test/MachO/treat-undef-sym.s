# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: not %lld -undefined bogus -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=UNKNOWN
# RUN: not %lld -undefined error -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=ERROR
# RUN:     %no_fatal_warnings_lld -undefined warning -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=WARNING
# RUN:     %lld -undefined suppress -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=SUPPRESS --allow-empty

# ERROR: error: undefined symbol: _bar
# ERROR-NEXT: >>> referenced by

# WARNING: warning: undefined symbol: _bar
# WARNING-NEXT: >>> referenced by

# SUPPRESS-NOT: undefined symbol: _bar

# UNKNOWN: unknown -undefined TREATMENT 'bogus'
# UNKNOWN-NEXT: error: undefined symbol: _bar
# UNKNOWN-NEXT: >>> referenced by

.globl _main
_main:
  callq _bar
  ret
