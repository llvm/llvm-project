# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/profile.proftext -o %t/profile.profdata

## The initializer is absent from the temporal profile, but dyld executes it
## before main. The opt-in initializer seed includes it in the startup order.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/baseline %t/input.o --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-compression-sort=none --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=BASELINE
# BASELINE: Ordered 1 sections (8 bytes) using balanced partitioning
# BASELINE: Functions for startup: 1 (8 bytes)

# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-startup-sort-initializers --bp-compression-sort=none --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=INITIALIZERS
# INITIALIZERS: Initializer functions for startup: 1
# INITIALIZERS: Ordered 2 sections (16 bytes) using balanced partitioning
# INITIALIZERS: Functions for startup: 2 (16 bytes)

# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-startup-sort-initializers --bp-compression-sort=none | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDER
# ORDER: _hot
# ORDER-NEXT: _initializer
# ORDER-NEXT: _main
# ORDER-NEXT: _cold

# RUN: not %lld -arch arm64 -lSystem -e _main -o /dev/null %t/input.o --bp-startup-sort-initializers 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: --bp-startup-sort-initializers must be used with --bp-startup-sort=function

#--- input.s
.subsections_via_symbols

.text
.globl _main
_main:
  ret

.globl _cold
_cold:
  mov w0, #1
  ret

.globl _hot
_hot:
  mov w0, #2
  ret

.globl _initializer
_initializer:
  mov w0, #3
  ret

.section __DATA,__mod_init_func,mod_init_funcs
.quad _initializer

#--- profile.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
hot,

hot
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1
