## Check that BOLT correctly adjusts Basic Block weights when given multiple indirect
## and direct calls with varying execution counts in the profile.

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.icl.fdata FDATA_INDIRECT_CALL_LARGER
# RUN: link_fdata %s %t.o %t.dcl.fdata FDATA_DIRECT_CALL_LARGER
# RUN: llvm-strip --strip-symbol=firstIndirectLabel %t.o
# RUN: llvm-strip --strip-symbol=secondIndirectLabel %t.o
# RUN: llvm-strip --strip-symbol=directLabel %t.o
# RUN: %clang %cflags64 -Wl,-q %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.icl.bolt --fix-block-counts --print-cfg \
# RUN:   --data=%t.icl.fdata 2>&1 | FileCheck %s --check-prefix=INDIRECT_HIGHER
# RUN: llvm-bolt %t.exe -o %t.dcl.bolt --fix-block-counts --print-cfg \
# RUN:   --data=%t.dcl.fdata 2>&1 | FileCheck %s --check-prefix=DIRECT_HIGHER

# FDATA_INDIRECT_CALL_LARGER: 0 [unknown] 0 1 main 0 0 0
# FDATA_INDIRECT_CALL_LARGER: 1 main #firstIndirectLabel# 1 foo 0 0 40
# FDATA_INDIRECT_CALL_LARGER: 1 main #firstIndirectLabel# 1 bar 0 0 20
# FDATA_INDIRECT_CALL_LARGER: 1 main #secondIndirectLabel# 1 foo 0 0 10
# FDATA_INDIRECT_CALL_LARGER: 1 main #secondIndirectLabel# 1 bar 0 0 30
# FDATA_INDIRECT_CALL_LARGER: 1 main #directLabel# 1 foo 0 0 20

# INDIRECT_HIGHER-LABEL: Binary Function "main" after building cfg
# INDIRECT_HIGHER: Entry Point
# INDIRECT_HIGHER-NEXT: Exec Count : 60{{$}}
# INDIRECT_HIGHER: jalr {{.*}}# CallProfile: 60 (0 misses) :
# INDIRECT_HIGHER-DAG: { foo: 40 (0 misses) }
# INDIRECT_HIGHER-DAG: { bar: 20 (0 misses) }
# INDIRECT_HIGHER: jalr {{.*}}# CallProfile: 40 (0 misses) :
# INDIRECT_HIGHER-DAG: { foo: 10 (0 misses) }
# INDIRECT_HIGHER-DAG: { bar: 30 (0 misses) }
# INDIRECT_HIGHER: jal {{.*}}foo # Count: 20

# FDATA_DIRECT_CALL_LARGER: 0 [unknown] 0 1 main 0 0 0
# FDATA_DIRECT_CALL_LARGER: 1 main #firstIndirectLabel# 1 foo 0 0 10
# FDATA_DIRECT_CALL_LARGER: 1 main #firstIndirectLabel# 1 bar 0 0 30
# FDATA_DIRECT_CALL_LARGER: 1 main #secondIndirectLabel# 1 foo 0 0 40
# FDATA_DIRECT_CALL_LARGER: 1 main #secondIndirectLabel# 1 bar 0 0 20
# FDATA_DIRECT_CALL_LARGER: 1 main #directLabel# 1 foo 0 0 100

# DIRECT_HIGHER-LABEL: Binary Function "main" after building cfg
# DIRECT_HIGHER: Entry Point
# DIRECT_HIGHER-NEXT: Exec Count : 100{{$}}
# DIRECT_HIGHER: {{.*}}# CallProfile: 40 (0 misses) :
# DIRECT_HIGHER-DAG: { bar: 30 (0 misses) }
# DIRECT_HIGHER-DAG: { foo: 10 (0 misses) }
# DIRECT_HIGHER: {{.*}}# CallProfile: 60 (0 misses) :
# DIRECT_HIGHER-DAG: { bar: 20 (0 misses) }
# DIRECT_HIGHER-DAG: { foo: 40 (0 misses) }
# DIRECT_HIGHER: jal {{.*}}foo # Count: 100

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
    lla a1, bar
firstIndirectLabel:
    jalr a0
secondIndirectLabel:
    jalr a1
directLabel:
    jal ra, foo
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
    .size main, .-main
