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

int internot(int a, int b, int c, int d, int e, int f) {
  return !(!(!a && b) || !(!!(!c && d) || !(e && !f)));
}

// MCDC-LABEL: @internot(
// MCDC-DAG:  store i32 0, ptr %mcdc.addr, align 4

// Branch #2, (#0 - #2) [1,3,0]
// !a [+0 => b][+6 => END]
// MCDC-DAG:  %[[A:.+]] = load i32, ptr %a.addr, align 4
// MCDC-DAG:  %[[AB:.+]] = icmp ne i32 %[[A]], 0
// MCDC-DAG:  %[[AN:.+]] = xor i1 %[[AB]], true
// MCDC-DAG:  %[[M1:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M1T:.+]] = add i32 %[[M1]], 0
// MCDC-DAG:  %[[M1F:.+]] = add i32 %[[M1]], 6
// MCDC-DAG:  %[[M1S:.+]] = select i1 %[[AN]], i32 %[[M1T]], i32 %[[M1F]]
// MCDC-DAG:  store i32 %[[M1S]], ptr %mcdc.addr, align 4

// Branch #3, (#2 - #3) [3,2,0]
// b [+0 => c][+7 => END]
// MCDC-DAG:  %[[B:.+]] = load i32, ptr %b.addr, align 4
// MCDC-DAG:  %[[BB:.+]] = icmp ne i32 %[[B]], 0
// MCDC-DAG:  %[[M3:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M3T:.+]] = add i32 %[[M3]], 0
// MCDC-DAG:  %[[M3F:.+]] = add i32 %[[M3]], 7
// MCDC-DAG:  %[[M3S:.+]] = select i1 %[[BB]], i32 %[[M3T]], i32 %[[M3F]]
// MCDC-DAG:  store i32 %[[M3S]], ptr %mcdc.addr, align 4

// Branch #5, (#1 - #5) [2,5,4]
// !c [+0 => d][+0 => e]
// MCDC-DAG:  %[[C:.+]] = load i32, ptr %c.addr, align 4
// MCDC-DAG:  %[[CB:.+]] = icmp ne i32 %[[C]], 0
// MCDC-DAG:  %[[CN:.+]] = xor i1 %[[CB]], true
// MCDC-DAG:  %[[M2:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M2T:.+]] = add i32 %[[M2]], 0
// MCDC-DAG:  %[[M2F:.+]] = add i32 %[[M2]], 0
// MCDC-DAG:  %[[M2S:.+]] = select i1 %[[CN]], i32 %[[M2T]], i32 %[[M2F]]
// MCDC-DAG:  store i32 %[[M2S]], ptr %mcdc.addr, align 4

// Branch #6, (#5 - #6) [5,0,4]
// d [+8 => END][+1 => e]]
// MCDC-DAG:  %[[D:.+]] = load i32, ptr %d.addr, align 4
// MCDC-DAG:  %[[DB:.+]] = icmp ne i32 %[[D]], 0
// MCDC-DAG:  %[[M5:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M5T:.+]] = add i32 %[[M5]], 8
// MCDC-DAG:  %[[M5F:.+]] = add i32 %[[M5]], 1
// MCDC-DAG:  %[[M5S:.+]] = select i1 %[[DB]], i32 %[[M5T]], i32 %[[M5F]]
// MCDC-DAG:  store i32 %[[M5S]], ptr %mcdc.addr, align 4

// Branch #7, (#4 - #7) [4,6,0]
// e [+0 => f][+0 => END]
// from:
//   [c => +0]
//   [d => +1]
// MCDC-DAG:  %[[E:.+]] = load i32, ptr %e.addr, align 4
// MCDC-DAG:  %[[EB:.+]] = icmp ne i32 %[[E]], 0
// MCDC-DAG:  %[[M4:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M4T:.+]] = add i32 %[[M4]], 0
// MCDC-DAG:  %[[M4F:.+]] = add i32 %[[M4]], 0
// MCDC-DAG:  %[[M4S:.+]] = select i1 %[[EB]], i32 %[[M4T]], i32 %[[M4F]]
// MCDC-DAG:  store i32 %[[M4S]], ptr %mcdc.addr, align 4

// Branch #8, (#7 - #8) [6,0,0]
// !f [+4 => END][+2 => END]
// MCDC-DAG:  %[[F:.+]] = load i32, ptr %f.addr, align 4
// MCDC-DAG:  %[[FB:.+]] = icmp ne i32 %[[F]], 0
// MCDC-DAG:  %[[FN:.+]] = xor i1 %[[FB]], true
// MCDC-DAG:  %[[M6:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[M6T:.+]] = add i32 %[[M6]], 4
// MCDC-DAG:  %[[M6F:.+]] = add i32 %[[M6]], 2
// MCDC-DAG:  %[[M6S:.+]] = select i1 %[[FN]], i32 %[[M6T]], i32 %[[M6F]]
// MCDC-DAG:  store i32 %[[M6S]], ptr %mcdc.addr, align 4

// from:
//   [e => +0]
//   [f => +2]
//     [c => +0]
//     [d => +1]
//   [f => +4]
//     [c => +0]
//     [d => +1]
//   [a => +6]
//   [b => +7]
//   [d => +8]
// MCDC-DAG:  %[[T0:.+]] = load i32, ptr %mcdc.addr, align 4
// MCDC-DAG:  %[[T:.+]] = add i32 %[[T0]], 0
// MCDC-DAG:  %[[TA:.+]] = lshr i32 %[[T]], 3
// MCDC-DAG:  %[[BA:.+]] = getelementptr inbounds i8, ptr @__profbm_internot, i32 %[[TA]]
// MCDC-DAG:  %[[BI:.+]] = and i32 %[[T]], 7
// MCDC-DAG:  %[[BI1:.+]] = trunc i32 %[[BI]] to i8
// MCDC-DAG:  %[[BM:.+]] = shl i8 1, %[[BI1]]
// MCDC-DAG:  %mcdc.bits = load i8, ptr %[[BA]], align 1
// MCDC-DAG:  %[[BN:.+]] = or i8 %mcdc.bits, %[[BM]]
// MCDC-DAG:  store i8 %[[BN]], ptr %[[BA]], align 1
