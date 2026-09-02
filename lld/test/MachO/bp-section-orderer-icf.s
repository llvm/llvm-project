# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/profile.proftext -o %t/profile.profdata

# RUN: %lld -arch arm64 -e _main -o %t/icf-all %t/input.o --icf=all -dead_strip --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=ALL-VERBOSE
# ALL-VERBOSE: Ordered 1 sections ({{.*}} bytes) using balanced partitioning
# ALL-VERBOSE: Functions for startup: 1 ({{.*}} bytes)

# RUN: llvm-nm --numeric-sort --format=just-symbols %t/icf-all | FileCheck %s --check-prefix=ORDER-ALL
# ORDER-ALL:      _hot_a
# ORDER-ALL-NEXT: _hot_b
# ORDER-ALL-NEXT: _hot_c
# ORDER-ALL-NEXT: _hot_d
# ORDER-ALL-DAG:  _main
# ORDER-ALL-DAG:  _cold

## Safe-thunk ICF retains the first member, _hot_a, as the shared body and
## replaces later address-significant members, _hot_b and _hot_c, with thunks.
## Since _hot_a already names offset zero, no extra internal target symbol is
## needed. The temporal profile names _hot_b, so balanced partitioning must
## order that thunk and its _hot_a body. The unprofiled _hot_c thunk must not be
## promoted.
# RUN: %lld -arch arm64 -e _main -o %t/icf-safe %t/input.o --icf=safe_thunks -dead_strip --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=SAFE-VERBOSE
# SAFE-VERBOSE: Ordered 3 sections ({{.*}} bytes) using balanced partitioning
# SAFE-VERBOSE: Functions for startup: 3 ({{.*}} bytes)

# RUN: llvm-nm --numeric-sort --format=just-symbols %t/icf-safe | FileCheck %s --check-prefix=ORDER-SAFE
# ORDER-SAFE:      _hot_a
# ORDER-SAFE-NEXT: _hot_b
# ORDER-SAFE-NEXT: _hot_c
# ORDER-SAFE-DAG:  _main
# ORDER-SAFE-DAG:  _cold
# ORDER-SAFE-DAG:  _hot_d

#--- input.s
.subsections_via_symbols

.addrsig
.addrsig_sym _hot_a
.addrsig_sym _hot_b
.addrsig_sym _hot_c
.addrsig_sym _hot_d

.text
.globl _main
_main:
  bl _hot_a
  bl _hot_b
  bl _hot_c
  bl _hot_d
  bl _cold
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

.globl _hot_d
_hot_d:
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
hot_b, hot_c

hot_b
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

hot_c
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1
