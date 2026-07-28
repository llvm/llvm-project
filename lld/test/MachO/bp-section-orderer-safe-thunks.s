# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/profile.proftext -o %t/profile.profdata

## The temporal profile names _hot_b, whose input section becomes a
## linker-created ICF thunk. Balanced partitioning must order both that thunk
## and the shared _hot_a body that it immediately branches to. _hot_c folds to
## another thunk for the same body but is not profiled and must not be promoted.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/out %t/input.o --icf=safe_thunks --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-compression-sort=none --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=VERBOSE
# VERBOSE: Ordered 2 sections (12 bytes) using balanced partitioning
# VERBOSE: Functions for startup: 2 (12 bytes)

# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --icf=safe_thunks --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --bp-compression-sort=none | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDER
# ORDER: _hot_a
# ORDER-NEXT: _hot_b
# ORDER-NEXT: _main
# ORDER-NEXT: _cold
# ORDER-NEXT: _hot_c

#--- input.s
.subsections_via_symbols

.addrsig
.addrsig_sym _hot_a
.addrsig_sym _hot_b
.addrsig_sym _hot_c

.text
.globl _main
_main:
  ret

.globl _hot_a
_hot_a:
  mov w0, #42
  ret

.globl _hot_b
_hot_b:
  mov w0, #42
  ret

.globl _hot_c
_hot_c:
  mov w0, #42
  ret

.globl _cold
_cold:
  mov w0, #1
  ret

#--- profile.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
hot_b

hot_b
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1
