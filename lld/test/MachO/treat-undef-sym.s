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
# RUN: %lld -undefined dynamic_lookup -lSystem -o %t.out %t.o 2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t.out \
# RUN:     | FileCheck --check-prefix=BIND %s

# RUN: %no_fatal_warnings_lld -lSystem -flat_namespace -undefined warning \
# RUN:     -o %t.out %t.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=WARNING
# RUN: llvm-objdump --macho --lazy-bind %t.out \
# RUN:     | FileCheck --check-prefix=BIND %s
# RUN: %lld -flat_namespace -lSystem -undefined suppress -o %t.out %t.o 2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t.out \
# RUN:     | FileCheck --check-prefix=BIND %s
# RUN: %lld -flat_namespace -lSystem -undefined dynamic_lookup -o %t.out %t.o 2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t.out \
# RUN:     | FileCheck --check-prefix=BIND %s

# ERROR: error: undefined symbol: _bar
# ERROR-NEXT: >>> referenced by

# INVAL-WARNING: error: '-undefined warning' only valid with '-flat_namespace'
# INVAL-WARNING-NEXT: error: undefined symbol: _bar

# INVAL-SUPPRESS: error: '-undefined suppress' only valid with '-flat_namespace'
# INVAL-SUPPRESS-NEXT: error: undefined symbol: _bar

# WARNING: warning: undefined symbol: _bar
# WARNING-NEXT: >>> referenced by

# UNKNOWN: unknown -undefined TREATMENT 'bogus'
# UNKNOWN-NEXT: error: undefined symbol: _bar
# UNKNOWN-NEXT: >>> referenced by

# BIND: Lazy bind table:
# BIND: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} flat-namespace   _bar

.globl _main
_main:
  callq _bar
  ret
