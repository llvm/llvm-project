// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc | FileCheck %s -check-prefix=MCDC
// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s -check-prefix=NOMCDC

int test(int b, int c, int d, int e, int f) {
  return ((b ? c : d) && e && f);
}

// NOMCDC-NOT: %mcdc.addr
// NOMCDC-NOT: __profbm_test

// MCDC BOOKKEEPING.
// MCDC: @__profbm_test = private global [1 x i8] zeroinitializer

// ALLOCATE MCDC TEMP AND ZERO IT.
// MCDC-LABEL: @test(
// MCDC: %mcdc.addr = alloca i32, align 4
// MCDC: store i32 0, ptr %mcdc.addr, align 4

// TERNARY TRUE SHOULD SHIFT ID = 0 FOR CONDITION 'c'.
// MCDC-LABEL: cond.true:
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %c.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDC-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 0
// MCDC-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// TERNARY FALSE SHOULD SHIFT ID = 0 FOR CONDITION 'd'.
// MCDC-LABEL: cond.false:
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %d.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDC-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 0
// MCDC-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT SECOND CONDITION WITH ID = 2.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %e.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDC-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 2
// MCDC-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT THIRD CONDITION WITH ID = 1.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %f.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDC-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 1
// MCDC-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// UPDATE FINAL BITMASK WITH RESULT.
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB4:[0-9]+]] = getelementptr inbounds i8, ptr @__profbm_test, i32 %[[LAB1]]
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %mcdc.bits = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB8:[0-9]+]] = or i8 %mcdc.bits, %[[LAB7]]
// MCDC:  store i8 %[[LAB8]], ptr %[[LAB4]], align 1
