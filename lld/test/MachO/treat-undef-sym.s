# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/live.s -o %t/live.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/dead.s -o %t/dead.o
        
# RUN: not %lld -undefined bogus -o /dev/null %t/live.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=UNKNOWN
# RUN: not %lld -undefined error -o /dev/null %t/live.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=ERROR

# RUN: not %lld -undefined warning -o /dev/null %t/live.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=INVAL-WARNING
# RUN: not %lld -undefined suppress -o /dev/null %t/live.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=INVAL-SUPPRESS
# RUN: %lld -undefined dynamic_lookup -lSystem -o %t/live.out %t/live.o 2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t/live.out \
# RUN:     | FileCheck --check-prefix=BIND %s

# RUN: %no_fatal_warnings_lld -lSystem -flat_namespace -undefined warning \
# RUN:     -o %t/live.out %t/live.o 2>&1 | \
# RUN:     FileCheck %s -check-prefix=WARNING
# RUN: llvm-objdump --macho --lazy-bind %t/live.out \
# RUN:     | FileCheck --check-prefix=BIND %s
# RUN: %lld -flat_namespace -lSystem -undefined suppress -o %t/live.out %t/live.o \
# RUN:     2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t/live.out \
# RUN:     | FileCheck --check-prefix=BIND %s
# RUN: %lld -flat_namespace -lSystem -undefined dynamic_lookup -o \
# RUN:     %t/live.out %t/live.o 2>&1 | count 0
# RUN: llvm-objdump --macho --lazy-bind %t/live.out \
# RUN:     | FileCheck --check-prefix=BIND %s

## Undefined symbols in dead code should not raise an error iff
## -dead_strip is enabled.
# RUN: not %lld -dylib -undefined error -o /dev/null %t/dead.o 2>&1 \
# RUN:     | FileCheck --check-prefix=ERROR %s
# RUN: not %lld -dylib -dead_strip -undefined error -o /dev/null %t/live.o 2>&1\
# RUN:     | FileCheck --check-prefix=ERROR %s
# RUN: %lld -dylib -dead_strip -undefined error -o /dev/null %t/dead.o

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

#--- live.s
.globl _main
_main:
  callq _bar
  ret

#--- dead.s
_dead:
  callq _bar
  ret
