// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc | FileCheck %s -check-prefix=MCDC
// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s -check-prefix=NOMCDC

int test(int a, int b, int c, int d, int e, int f) {
  return ((!a && b) || ((!c && d) || (e && !f)));
}

// NOMCDC-NOT: %mcdc.addr
// NOMCDC-NOT: __profbm_test

// MCDC BOOKKEEPING.
// MCDC: @__profbm_test = private global [2 x i8] zeroinitializer
// MCDC: @__profc_test = private global [9 x i64] zeroinitializer

// ALLOCATE MCDC TEMP AND ZERO IT.
// MCDC-LABEL: @test(
// MCDC: %mcdc.addr = alloca i32, align 4
// MCDC: store i32 0, ptr %mcdc.addr, align 4

// SHIFT FIRST CONDITION WITH ID = 0.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %a.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[LNOT:lnot[0-9]*]] = xor i1 %[[BOOL]]
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[LNOT]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT SECOND CONDITION WITH ID = 2.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %b.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 14
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 1
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[BOOL]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT THIRD CONDITION WITH ID = 1.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %c.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[LNOT:lnot[0-9]*]] = xor i1 %[[BOOL]]
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[LNOT]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT FOURTH CONDITION WITH ID = 4.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %d.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 12
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 2
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[BOOL]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT FIFTH CONDITION WITH ID = 3.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %e.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 0
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[BOOL]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT SIXTH CONDITION WITH ID = 5.
// MCDC:  %[[LAB1:[0-9]+]] = load i32, ptr %f.addr, align 4
// MCDC-DAG:  %[[BOOL:tobool[0-9]*]] = icmp ne i32 %[[LAB1]], 0
// MCDC-DAG:  %[[LNOT:lnot[0-9]*]] = xor i1 %[[BOOL]]
// MCDC-DAG:  %[[TEMP:mcdc.*]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[LAB2:[0-9]+]] = add i32 %[[TEMP]], 8
// MCDC-DAG:  %[[LAB3:[0-9]+]] = add i32 %[[TEMP]], 4
// MCDC-DAG:  %[[LAB4:[0-9]+]] = select i1 %[[LNOT]], i32 %[[LAB2]], i32 %[[LAB3]]
// MCDC-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// UPDATE FINAL BITMASK WITH RESULT.
// MCDC-DAG:  %[[TEMP0:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[TEMP:[0-9]+]] = add i32 %[[TEMP0]], 0
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB4:[0-9]+]] = getelementptr inbounds i8, ptr @__profbm_test, i32 %[[LAB1]]
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %[[BITS:.+]] = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB8:[0-9]+]] = or i8 %[[BITS]], %[[LAB7]]
// MCDC:  store i8 %[[LAB8]], ptr %[[LAB4]], align 1
