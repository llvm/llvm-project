## Check that BOLT correctly adjusts Basic Block weights when given indirect calls
## and a variety of different execution counts in the profile.

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.zcc.fdata FDATA_ZERO_BB_COUNT
# RUN: link_fdata %s %t.o %t.bcc.fdata FDATA_BELOW_CALL_COUNT
# RUN: link_fdata %s %t.o %t.acc.fdata FDATA_ABOVE_CALL_COUNT
# RUN: llvm-strip --strip-symbol=callLabel %t.o
# RUN: %clang %cflags64 -Wl,-q %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.zcc.bolt --fix-block-counts --print-cfg \
# RUN:   --data=%t.zcc.fdata 2>&1 | FileCheck %s --check-prefix=ZERO
# RUN: llvm-bolt %t.exe -o %t.bcc.bolt --fix-block-counts --print-cfg \
# RUN:   --data=%t.bcc.fdata 2>&1 | FileCheck %s --check-prefix=BELOW
# RUN: llvm-bolt %t.exe -o %t.acc.bolt --fix-block-counts --print-cfg \
# RUN:   --data=%t.acc.fdata 2>&1 | FileCheck %s --check-prefix=ABOVE

# FDATA_ZERO_BB_COUNT: 0 [unknown] 0 1 main 0 0 0
# FDATA_ZERO_BB_COUNT: 1 main #callLabel# 1 foo 0 0 20
# FDATA_ZERO_BB_COUNT: 1 main #callLabel# 1 bar 0 0 30

# ZERO-LABEL: Binary Function "main" after building cfg
# ZERO: Entry Point
# ZERO-NEXT: Exec Count : 50{{$}}
# ZERO: jalr {{.*}}# CallProfile: 50 (0 misses) :
# ZERO-DAG: { foo: 20 (0 misses) }
# ZERO-DAG: { bar: 30 (0 misses) }

# FDATA_BELOW_CALL_COUNT: 0 [unknown] 0 1 main 0 0 10
# FDATA_BELOW_CALL_COUNT: 1 main #callLabel# 1 foo 0 0 30
# FDATA_BELOW_CALL_COUNT: 1 main #callLabel# 1 bar 0 0 70

# BELOW-LABEL: Binary Function "main" after building cfg
# BELOW: Entry Point
# BELOW-NEXT: Exec Count : 100{{$}}
# BELOW: jalr {{.*}}# CallProfile: 100 (0 misses) :
# BELOW-DAG: { foo: 30 (0 misses) }
# BELOW-DAG: { bar: 70 (0 misses) }

# FDATA_ABOVE_CALL_COUNT: 0 [unknown] 0 1 main 0 0 90
# FDATA_ABOVE_CALL_COUNT: 1 main #callLabel# 1 foo 0 0 20
# FDATA_ABOVE_CALL_COUNT: 1 main #callLabel# 1 bar 0 0 30

# ABOVE-LABEL: Function "main" after building cfg
# ABOVE: Entry Point
# ABOVE-NEXT: Exec Count : 90{{$}}
# ABOVE: jalr {{.*}}# CallProfile: 50 (0 misses) :
# ABOVE-DAG: { foo: 20 (0 misses) }
# ABOVE-DAG: { bar: 30 (0 misses) }

        .type main,@function
        .type foo,@function
        .type bar,@function
        .globl main,foo,bar
foo:
    ret
    .size foo, .-foo

bar:
    ret
    .size bar, .-bar

main:
    addi sp, sp, -16
    sd ra, 8(sp)
    lla a0, foo
callLabel:
    jalr a0
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
    .size main, .-main
