# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata

## Compression sort only: all non-cold functions should appear before all cold
## ones, despite input order interleaving them.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/compr.out %t/a.o --bp-compression-sort=function
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/compr.out | FileCheck %s --check-prefix=COMPRESSION

# COMPRESSION:         _main
# COMPRESSION:         _hot1
# COMPRESSION:         _hot2
# COMPRESSION:         _hot3
# COMPRESSION:         _cold1
# COMPRESSION:         _cold2
# COMPRESSION:         _cold3

## Startup sort only: _hot1 and _cold1 are in the startup trace and get ordered
## first. Non-startup non-cold sections keep input order, then non-startup cold
## sections are pushed to the end.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/startup-only.out %t/a.o --irpgo-profile=%t/a.profdata --bp-startup-sort=function --verbose-bp-section-orderer 2> %t/startup-only-verbose.txt
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/startup-only.out | FileCheck %s --check-prefix=STARTUP-ONLY
# RUN: FileCheck %s --input-file %t/startup-only-verbose.txt --check-prefix=STARTUP-ONLY-VERBOSE

# STARTUP-ONLY:         _hot1
# STARTUP-ONLY:         _cold1
# STARTUP-ONLY:         _main
# STARTUP-ONLY:         _hot2
# STARTUP-ONLY:         _hot3
# STARTUP-ONLY:         _cold2
# STARTUP-ONLY:         _cold3
# STARTUP-ONLY-VERBOSE: Functions for startup: 2
# STARTUP-ONLY-VERBOSE: Functions for compression: 0
# STARTUP-ONLY-VERBOSE: Cold functions for compression: 0

## Startup sort + compression sort: startup functions first, then non-cold
## functions, then cold functions.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/startup-compr.out %t/a.o --irpgo-profile=%t/a.profdata --bp-startup-sort=function --bp-compression-sort=function --verbose-bp-section-orderer 2> %t/startup-compr-verbose.txt
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/startup-compr.out | FileCheck %s --check-prefix=STARTUP-COMPR
# RUN: FileCheck %s --input-file %t/startup-compr-verbose.txt --check-prefix=STARTUP-COMPR-VERBOSE

# STARTUP-COMPR:         _hot1
# STARTUP-COMPR:         _cold1
# STARTUP-COMPR:         _main
# STARTUP-COMPR:         _hot2
# STARTUP-COMPR:         _hot3
# STARTUP-COMPR:         _cold2
# STARTUP-COMPR:         _cold3
# STARTUP-COMPR-VERBOSE: Functions for startup: 2
# STARTUP-COMPR-VERBOSE: Functions for compression: 3
# STARTUP-COMPR-VERBOSE: Cold functions for compression: 2

## Order file takes precedence over BP ordering. A cold function in the order
## file appears at its ordered position, not in the cold region.
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/order.out %t/a.o --bp-compression-sort=function -order_file %t/a.orderfile
# RUN: llvm-nm --numeric-sort --format=just-symbols %t/order.out | FileCheck %s --check-prefix=ORDERFILE

# ORDERFILE:     _cold2
# ORDERFILE:     _hot1

#--- a.s
.subsections_via_symbols
.text

.globl _main
_main:
  ret

.globl _cold1
.desc _cold1, 0x400
_cold1:
  add w0, w0, #10
  add w1, w1, #11
  bl _main
  ret

.globl _hot1
_hot1:
  add w0, w0, #1
  add w1, w1, #2
  bl _main
  ret

.globl _cold2
.desc _cold2, 0x400
_cold2:
  add w0, w0, #20
  add w1, w1, #21
  bl _hot1
  ret

.globl _hot2
_hot2:
  add w0, w0, #2
  add w1, w1, #3
  bl _hot1
  ret

.globl _cold3
.desc _cold3, 0x400
_cold3:
  add w0, w0, #30
  add w1, w1, #31
  bl _cold1
  ret

.globl _hot3
_hot3:
  add w0, w0, #3
  add w1, w1, #4
  bl _cold1
  ret

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
hot1, cold1

hot1
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

cold1
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

#--- a.orderfile
_cold2
_hot1
