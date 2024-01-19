// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc | FileCheck %s -check-prefix=MCDC
// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s -check-prefix=NOMCDC

int test(int a, int b, int c, int d, int e, int f) {
  return ((a || b) ? (c && d) : (e || f));
}

// NOMCDC-NOT: %mcdc.addr
// NOMCDC-NOT: __profbm_test

// MCDC BOOKKEEPING.
// MCDC: @__profbm_test = private global [3 x i8] zeroinitializer

// ALLOCATE MCDC TEMP AND ZERO IT.
// MCDC-LABEL: @test(
// MCDC: %mcdc.addr = alloca i32, align 4
// MCDC: store i32 0, ptr %mcdc.addr, align 4

// TERNARY TRUE SHOULD UPDATE THE BITMAP WITH RESULT AT ELEMENT 0.
// MCDC-LABEL: cond.true:
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB2:[0-9]+]] = zext i32 %[[LAB1]] to i64
// MCDC:  %[[LAB3:[0-9]+]] = add i64 ptrtoint (ptr @__profbm_test to i64), %[[LAB2]]
// MCDC:  %[[LAB4:[0-9]+]] = inttoptr i64 %[[LAB3]] to ptr
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %[[LAB8:mcdc.bits[0-9]*]] = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB9:[0-9]+]] = or i8 %[[LAB8]], %[[LAB7]]
// MCDC:  store i8 %[[LAB9]], ptr %[[LAB4]], align 1

// CHECK FOR ZERO OF MCDC TEMP
// MCDC: store i32 0, ptr %mcdc.addr, align 4

// TERNARY TRUE YIELDS TERNARY LHS LOGICAL-AND.
// TERNARY LHS LOGICAL-AND SHOULD UPDATE THE BITMAP WITH RESULT AT ELEMENT 1.
// MCDC-LABEL: land.end:
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB2:[0-9]+]] = zext i32 %[[LAB1]] to i64
// MCDC:  %[[LAB3:[0-9]+]] = add i64 ptrtoint (ptr getelementptr inbounds ([3 x i8], ptr @__profbm_test, i32 0, i32 1) to i64), %[[LAB2]]
// MCDC:  %[[LAB4:[0-9]+]] = inttoptr i64 %[[LAB3]] to ptr
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %[[LAB8:mcdc.bits[0-9]*]] = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB9:[0-9]+]] = or i8 %[[LAB8]], %[[LAB7]]
// MCDC:  store i8 %[[LAB9]], ptr %[[LAB4]], align 1

// TERNARY FALSE SHOULD UPDATE THE BITMAP WITH RESULT AT ELEMENT 0.
// MCDC-LABEL: cond.false:
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB2:[0-9]+]] = zext i32 %[[LAB1]] to i64
// MCDC:  %[[LAB3:[0-9]+]] = add i64 ptrtoint (ptr @__profbm_test to i64), %[[LAB2]]
// MCDC:  %[[LAB4:[0-9]+]] = inttoptr i64 %[[LAB3]] to ptr
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %[[LAB8:mcdc.bits[0-9]*]] = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB9:[0-9]+]] = or i8 %[[LAB8]], %[[LAB7]]
// MCDC:  store i8 %[[LAB9]], ptr %[[LAB4]], align 1

// CHECK FOR ZERO OF MCDC TEMP
// MCDC: store i32 0, ptr %mcdc.addr, align 4

// TERNARY FALSE YIELDS TERNARY RHS LOGICAL-OR.
// TERNARY RHS LOGICAL-OR SHOULD UPDATE THE BITMAP WITH RESULT AT ELEMENT 2.
// MCDC-LABEL: lor.end:
// MCDC-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDC:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDC:  %[[LAB2:[0-9]+]] = zext i32 %[[LAB1]] to i64
// MCDC:  %[[LAB3:[0-9]+]] = add i64 ptrtoint (ptr getelementptr inbounds ([3 x i8], ptr @__profbm_test, i32 0, i32 2) to i64), %[[LAB2]]
// MCDC:  %[[LAB4:[0-9]+]] = inttoptr i64 %[[LAB3]] to ptr
// MCDC:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDC:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDC:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDC:  %[[LAB8:mcdc.bits[0-9]*]] = load i8, ptr %[[LAB4]], align 1
// MCDC:  %[[LAB9:[0-9]+]] = or i8 %[[LAB8]], %[[LAB7]]
// MCDC:  store i8 %[[LAB9]], ptr %[[LAB4]], align 1
