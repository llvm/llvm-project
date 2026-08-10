# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/profile.proftext -o %t/profile.profdata

## Safe-thunk ICF retains the first member, _hot_a, as the shared body and
## replaces later address-significant members, _hot_b and _hot_c, with thunks.
## Since _hot_a already names offset zero, no extra internal target symbol is
## needed. The temporal profile names _hot_b, so balanced partitioning must
## order that thunk and its _hot_a body. The unprofiled _hot_c thunk must not be
## promoted.
# RUN: %lld -arch arm64 -e _main -o %t/out %t/input.o --icf=safe_thunks --irpgo-profile=%t/profile.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=VERBOSE
# VERBOSE: Ordered 2 sections (12 bytes) using balanced partitioning
# VERBOSE: Functions for startup: 2 (12 bytes)

# RUN: llvm-objdump --no-show-raw-insn -d %t/out | FileCheck %s --check-prefix=THUNKS
# THUNKS-LABEL: <_hot_a>:
# THUNKS-NEXT: {{.*}} mov w0, #0x2a
# THUNKS-NEXT: {{.*}} ret
# THUNKS-LABEL: <_hot_b>:
# THUNKS-NEXT: {{.*}} b {{.*}} <_hot_a>
# THUNKS-LABEL: <_hot_c>:
# THUNKS-NEXT: {{.*}} b {{.*}} <_hot_a>

# RUN: %lld -arch arm64 -e _main -o - %t/input.o --icf=safe_thunks --irpgo-profile=%t/profile.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDER
# ORDER: _hot_a
# ORDER-NEXT: _hot_b
# ORDER-DAG: _main
# ORDER-DAG: _cold
# ORDER-DAG: _hot_c

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
