# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/weighted.proftext %t/functions.proftext -o %t/weighted.profdata
# RUN: llvm-profdata merge %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext \
# RUN:   %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext \
# RUN:   %t/competing.proftext %t/functions.proftext -o %t/replicated.profdata
# RUN: llvm-profdata merge %t/primary.proftext %t/competing.proftext %t/functions.proftext -o %t/unweighted.profdata
# RUN: llvm-profdata merge %t/saturated.proftext %t/functions.proftext -o %t/saturated.profdata

# A weight of 10 is equivalent to ten copies of the same trace.
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - > %t/weighted.order
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - > %t/replicated.order
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/unweighted.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - > %t/unweighted.order
# RUN: cmp %t/weighted.order %t/replicated.order
# RUN: not cmp %t/weighted.order %t/unweighted.order
# RUN: FileCheck %s --input-file=%t/weighted.order --check-prefix=WEIGHTED
# RUN: FileCheck %s --input-file=%t/unweighted.order --check-prefix=UNWEIGHTED

# WEIGHTED: D
# WEIGHTED-NEXT: A
# WEIGHTED-NEXT: B
# WEIGHTED-NEXT: C
# WEIGHTED-NEXT: E
# WEIGHTED-NEXT: F
# UNWEIGHTED: A
# UNWEIGHTED-NEXT: B
# UNWEIGHTED-NEXT: D
# UNWEIGHTED-NEXT: C
# UNWEIGHTED-NEXT: E
# UNWEIGHTED-NEXT: F

# Combining weight-one compression utilities with weighted temporal utilities
# preserves the same weighted-versus-replicated equivalence.
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function --bp-compression-sort-startup-functions | llvm-nm --numeric-sort --format=just-symbols - > %t/weighted-compression.order
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function --bp-compression-sort-startup-functions | llvm-nm --numeric-sort --format=just-symbols - > %t/replicated-compression.order
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/unweighted.profdata --bp-startup-sort=function --bp-compression-sort-startup-functions | llvm-nm --numeric-sort --format=just-symbols - > %t/unweighted-compression.order
# RUN: cmp %t/weighted-compression.order %t/replicated-compression.order
# RUN: not cmp %t/weighted-compression.order %t/unweighted-compression.order

# Verbose trace-reference counts and the page-fault area also respect weights.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/weighted.out %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=WEIGHTED-STATS
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/replicated.out %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=WEIGHTED-STATS
# WEIGHTED-STATS: Temporal profile function references: 66 / 66 resolved (6 / 6 unique)
# WEIGHTED-STATS: Total area under the page fault curve: 6.600000e+01

# Oversized weights saturate verbose diagnostics instead of wrapping.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/saturated.out %t/input.o --irpgo-profile=%t/saturated.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=SATURATED-STATS
# SATURATED-STATS: Temporal profile function references: 18446744073709551615 / 18446744073709551615 resolved (2 / 2 unique)
# SATURATED-STATS: Total area under the page fault curve: 1.844674e+19

#--- input.s
.text
.globl _main, A, B, C, D, E, F
_main:
  ret
A:
  add w0, w0, #1
  ret
B:
  add w0, w0, #1
  ret
C:
  add w0, w0, #2
  ret
D:
  add w0, w0, #2
  ret
E:
  add w0, w0, #3
  ret
F:
  add w0, w0, #3
  ret
.subsections_via_symbols

#--- weighted.proftext
:ir
:temporal_prof_traces
# Num Traces
2
# Trace Stream Size
11
# Weight
10
A, B, C, D, E, F
# Weight
1
A, D, B, C, E, F

#--- primary.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size
1
# Weight
1
A, B, C, D, E, F

#--- competing.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size
1
# Weight
1
A, D, B, C, E, F

#--- saturated.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size
1
# Weight
18446744073709551615
A, B

#--- functions.proftext
:ir
A
# Func Hash
1
# Num Counters
1
# Counter Values
1
B
2
1
1
C
3
1
1
D
4
1
1
E
5
1
1
F
6
1
1
