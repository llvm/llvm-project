# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: not %lld -undefined bogus -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=UNKNOWN
# RUN: not %lld -undefined error -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=ERROR

# RUN: not %lld -undefined warning -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=INVAL-WARNING
# RUN: not %lld -undefined suppress -o /dev/null %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=INVAL-SUPPRESS

# FIXME: Enable these -undefined checks once -flat_namespace is implemented.
# RN: %no_fatal_warnings_lld -flat_namespace -undefined warning \
# RN:     -o /dev/null %t.o 2>&1 | \
# RN:     FileCheck %s -check-prefix=WARNING
# RN: %lld -flat_namespace -undefined suppress -o /dev/null %t.o 2>&1 | \
# RN:     FileCheck %s -check-prefix=SUPPRESS --allow-empty

# ERROR: error: undefined symbol: _bar
# ERROR-NEXT: >>> referenced by

# INVAL-WARNING: error: '-undefined warning' only valid with '-flat_namespace'
# INVAL-WARNING-NEXT: error: undefined symbol: _bar

# INVAL-SUPPRESS: error: '-undefined suppress' only valid with '-flat_namespace'
# INVAL-SUPPRESS-NEXT: error: undefined symbol: _bar


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
