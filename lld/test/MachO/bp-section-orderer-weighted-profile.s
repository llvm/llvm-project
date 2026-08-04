# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/weighted.proftext %t/functions.proftext -o %t/weighted.profdata
# RUN: llvm-profdata merge %t/replicated.proftext %t/functions.proftext -o %t/replicated.profdata
# RUN: llvm-profdata merge %t/unweighted.proftext %t/functions.proftext -o %t/unweighted.profdata

# A weight of 10 is equivalent to ten copies of the same trace.
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=WEIGHTED
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=WEIGHTED
# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/input.o --irpgo-profile=%t/unweighted.profdata --bp-startup-sort=function | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=UNWEIGHTED

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

# Verbose trace-reference counts and the page-fault area also respect weights.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/weighted.out %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=WEIGHTED-STATS
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/replicated.out %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=WEIGHTED-STATS
# WEIGHTED-STATS: Temporal profile function references: 66 / 66 resolved (6 / 6 unique)
# WEIGHTED-STATS: Total area under the page fault curve: 6.600000e+01

#--- input.s
.text
.globl _main, A, B, C, D, E, F
_main:
  ret
A:
  ret
B:
  ret
C:
  ret
D:
  ret
E:
  ret
F:
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

#--- replicated.proftext
:ir
:temporal_prof_traces
# Num Traces
11
# Trace Stream Size
11
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, B, C, D, E, F
# Weight
1
A, D, B, C, E, F

#--- unweighted.proftext
:ir
:temporal_prof_traces
# Num Traces
2
# Trace Stream Size
2
# Weight
1
A, B, C, D, E, F
# Weight
1
A, D, B, C, E, F

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
