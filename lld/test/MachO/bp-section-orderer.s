# REQUIRES: aarch64, asserts

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/a.out %t/a.o --profile-guided-function-order=%t/a.profdata 2>&1 | FileCheck %s --check-prefix=STARTUP
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/a.out %t/a.o --profile-guided-function-order=%t/a.profdata --icf=all 2>&1 | FileCheck %s --check-prefix=STARTUP

# RUN: %lld -arch arm64 -lSystem -e _main -o - %t/a.o --profile-guided-function-order=%t/a.profdata -order_file %t/a.orderfile | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDERFILE

# RUN: %lld -arch arm64 -lSystem -e _main -o %t/a.out %t/a.o --function-order-for-compression --data-order-for-compression 2>&1 | FileCheck %s --check-prefix=COMPRESSION
# RUN: %lld -arch arm64 -lSystem -e _main -o %t/a.out %t/a.o --profile-guided-function-order=%t/a.profdata --function-order-for-compression --data-order-for-compression 2>&1 | FileCheck %s --check-prefix=COMPRESSION


# STARTUP: Ordered 3 sections using balanced partitioning

# ORDERFILE: A
# ORDERFILE: F
# ORDERFILE: E
# ORDERFILE: D
# ORDERFILE-DAG: _B
# ORDERFILE-DAG: l_C

# COMPRESSION: Ordered 11 sections using balanced partitioning

#--- a.s
.text
.globl _main, A, _B, l_C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222

_main:
  ret
A:
  ret
_B:
  add w0, w0, #1
  bl  A
  ret
l_C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222:
  add w0, w0, #2
  bl  A
  ret
D:
  add w0, w0, #2
  bl _B
  ret
E:
  add w0, w0, #2
  bl l_C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222
  ret
F:
  add w0, w0, #3
  bl l_C.__uniq.111111111111111111111111111111111111111.llvm.2222222222222222222
  ret

.data
s1:
  .ascii "hello world"
s2:
  .ascii "i am a string"
r1:
  .quad s1
r2:
  .quad r1

.subsections_via_symbols

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
A, B, C.__uniq.555555555555555555555555555555555555555.llvm.6666666666666666666

A
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

B
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

C.__uniq.555555555555555555555555555555555555555.llvm.6666666666666666666
# Func Hash:
3333
# Num Counters:
1
# Counter Values:
1

D
# Func Hash:
4444
# Num Counters:
1
# Counter Values:
1

#--- a.orderfile
A
F
E
D
