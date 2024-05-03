// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck -check-prefix=CHECK32 %s

// CHECK-LABEL: define dso_local x86_fp80 @testinc(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = atomicrmw fadd ptr [[TMP0]], float 1.000000e+00 seq_cst, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = fadd float [[TMP1]], 1.000000e+00
// CHECK-NEXT:    store float [[TMP2]], ptr [[RETVAL]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[RETVAL]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP3]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @testinc(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP1:%.*]] = atomicrmw fadd ptr [[TMP0]], float 1.000000e+00 seq_cst, align 4
// CHECK32-NEXT:    [[TMP2:%.*]] = fadd float [[TMP1]], 1.000000e+00
// CHECK32-NEXT:    store float [[TMP2]], ptr [[RETVAL]], align 4
// CHECK32-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[RETVAL]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP3]]
//
long double testinc(_Atomic long double *addr) {

  return ++*addr;
}

// CHECK-LABEL: define dso_local x86_fp80 @testdec(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = atomicrmw fsub ptr [[TMP0]], float 1.000000e+00 seq_cst, align 16
// CHECK-NEXT:    store float [[TMP1]], ptr [[RETVAL]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[RETVAL]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP2]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @testdec(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP1:%.*]] = atomicrmw fsub ptr [[TMP0]], float 1.000000e+00 seq_cst, align 4
// CHECK32-NEXT:    store float [[TMP1]], ptr [[RETVAL]], align 4
// CHECK32-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[RETVAL]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP2]]
//
long double testdec(_Atomic long double *addr) {

  return (*addr)--;
}

// CHECK-LABEL: define dso_local x86_fp80 @testcompassign(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP2:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP3:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP5:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD:%.*]] = load atomic i128, ptr [[TMP0]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD]], ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    br label [[ATOMIC_OP:%.*]]
// CHECK:       atomic_op:
// CHECK-NEXT:    [[TMP2:%.*]] = phi x86_fp80 [ [[TMP1]], [[ENTRY:%.*]] ], [ [[TMP8:%.*]], [[ATOMIC_OP]] ]
// CHECK-NEXT:    [[SUB:%.*]] = fsub x86_fp80 [[TMP2]], 0xK4003C800000000000000
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP1]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 [[TMP2]], ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load i128, ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP2]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 [[SUB]], ptr [[ATOMIC_TEMP2]], align 16
// CHECK-NEXT:    [[TMP4:%.*]] = load i128, ptr [[ATOMIC_TEMP2]], align 16
// CHECK-NEXT:    [[TMP5:%.*]] = cmpxchg ptr [[TMP0]], i128 [[TMP3]], i128 [[TMP4]] seq_cst seq_cst, align 16
// CHECK-NEXT:    [[TMP6:%.*]] = extractvalue { i128, i1 } [[TMP5]], 0
// CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { i128, i1 } [[TMP5]], 1
// CHECK-NEXT:    store i128 [[TMP6]], ptr [[ATOMIC_TEMP3]], align 16
// CHECK-NEXT:    [[TMP8]] = load x86_fp80, ptr [[ATOMIC_TEMP3]], align 16
// CHECK-NEXT:    br i1 [[TMP7]], label [[ATOMIC_CONT:%.*]], label [[ATOMIC_OP]]
// CHECK:       atomic_cont:
// CHECK-NEXT:    [[TMP9:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD4:%.*]] = load atomic i128, ptr [[TMP9]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD4]], ptr [[ATOMIC_TEMP5]], align 16
// CHECK-NEXT:    [[TMP10:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP5]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP10]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @testcompassign(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP2:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP3:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP1:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP]], align 4
// CHECK32-NEXT:    br label [[ATOMIC_OP:%.*]]
// CHECK32:       atomic_op:
// CHECK32-NEXT:    [[TMP2:%.*]] = phi x86_fp80 [ [[TMP1]], [[ENTRY:%.*]] ], [ [[TMP3:%.*]], [[ATOMIC_OP]] ]
// CHECK32-NEXT:    [[SUB:%.*]] = fsub x86_fp80 [[TMP2]], 0xK4003C800000000000000
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP1]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 [[TMP2]], ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP2]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 [[SUB]], ptr [[ATOMIC_TEMP2]], align 4
// CHECK32-NEXT:    [[CALL:%.*]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP1]], ptr noundef [[ATOMIC_TEMP2]], i32 noundef 5, i32 noundef 5)
// CHECK32-NEXT:    [[TMP3]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    br i1 [[CALL]], label [[ATOMIC_CONT:%.*]], label [[ATOMIC_OP]]
// CHECK32:       atomic_cont:
// CHECK32-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP4]], ptr noundef [[ATOMIC_TEMP3]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP5:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP3]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP5]]
//
long double testcompassign(_Atomic long double *addr) {
  *addr -= 25;
  return *addr;
}

// CHECK-LABEL: define dso_local x86_fp80 @testassign(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 0xK4005E600000000000000, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load i128, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    store atomic i128 [[TMP1]], ptr [[TMP0]] seq_cst, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD:%.*]] = load atomic i128, ptr [[TMP2]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD]], ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP3]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @testassign(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 0xK4005E600000000000000, ptr [[ATOMIC_TEMP]], align 4
// CHECK32-NEXT:    call void @__atomic_store(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP1]], ptr noundef [[ATOMIC_TEMP1]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP2]]
//
long double testassign(_Atomic long double *addr) {
  *addr = 115;

  return *addr;
}

// CHECK-LABEL: define dso_local x86_fp80 @test_volatile_inc(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = atomicrmw fadd ptr [[TMP0]], float 1.000000e+00 seq_cst, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = fadd float [[TMP1]], 1.000000e+00
// CHECK-NEXT:    store float [[TMP2]], ptr [[RETVAL]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[RETVAL]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP3]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @test_volatile_inc(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP1:%.*]] = atomicrmw fadd ptr [[TMP0]], float 1.000000e+00 seq_cst, align 4
// CHECK32-NEXT:    [[TMP2:%.*]] = fadd float [[TMP1]], 1.000000e+00
// CHECK32-NEXT:    store float [[TMP2]], ptr [[RETVAL]], align 4
// CHECK32-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[RETVAL]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP3]]
//
long double test_volatile_inc(volatile _Atomic long double *addr) {
  return ++*addr;
}

// CHECK-LABEL: define dso_local x86_fp80 @test_volatile_dec(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = atomicrmw fsub ptr [[TMP0]], float 1.000000e+00 seq_cst, align 16
// CHECK-NEXT:    store float [[TMP1]], ptr [[RETVAL]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[RETVAL]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP2]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @test_volatile_dec(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[RETVAL:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP1:%.*]] = atomicrmw fsub ptr [[TMP0]], float 1.000000e+00 seq_cst, align 4
// CHECK32-NEXT:    store float [[TMP1]], ptr [[RETVAL]], align 4
// CHECK32-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[RETVAL]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP2]]
//
long double test_volatile_dec(volatile _Atomic long double *addr) {
  return (*addr)--;
}

// CHECK-LABEL: define dso_local x86_fp80 @test_volatile_compassign(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP2:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP3:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP5:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD:%.*]] = load atomic volatile i128, ptr [[TMP0]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD]], ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    br label [[ATOMIC_OP:%.*]]
// CHECK:       atomic_op:
// CHECK-NEXT:    [[TMP2:%.*]] = phi x86_fp80 [ [[TMP1]], [[ENTRY:%.*]] ], [ [[TMP8:%.*]], [[ATOMIC_OP]] ]
// CHECK-NEXT:    [[SUB:%.*]] = fsub x86_fp80 [[TMP2]], 0xK4003C800000000000000
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP1]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 [[TMP2]], ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load i128, ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP2]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 [[SUB]], ptr [[ATOMIC_TEMP2]], align 16
// CHECK-NEXT:    [[TMP4:%.*]] = load i128, ptr [[ATOMIC_TEMP2]], align 16
// CHECK-NEXT:    [[TMP5:%.*]] = cmpxchg volatile ptr [[TMP0]], i128 [[TMP3]], i128 [[TMP4]] seq_cst seq_cst, align 16
// CHECK-NEXT:    [[TMP6:%.*]] = extractvalue { i128, i1 } [[TMP5]], 0
// CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { i128, i1 } [[TMP5]], 1
// CHECK-NEXT:    store i128 [[TMP6]], ptr [[ATOMIC_TEMP3]], align 16
// CHECK-NEXT:    [[TMP8]] = load x86_fp80, ptr [[ATOMIC_TEMP3]], align 16
// CHECK-NEXT:    br i1 [[TMP7]], label [[ATOMIC_CONT:%.*]], label [[ATOMIC_OP]]
// CHECK:       atomic_cont:
// CHECK-NEXT:    [[TMP9:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD4:%.*]] = load atomic volatile i128, ptr [[TMP9]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD4]], ptr [[ATOMIC_TEMP5]], align 16
// CHECK-NEXT:    [[TMP10:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP5]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP10]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @test_volatile_compassign(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP2:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP3:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP1:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP]], align 4
// CHECK32-NEXT:    br label [[ATOMIC_OP:%.*]]
// CHECK32:       atomic_op:
// CHECK32-NEXT:    [[TMP2:%.*]] = phi x86_fp80 [ [[TMP1]], [[ENTRY:%.*]] ], [ [[TMP3:%.*]], [[ATOMIC_OP]] ]
// CHECK32-NEXT:    [[SUB:%.*]] = fsub x86_fp80 [[TMP2]], 0xK4003C800000000000000
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP1]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 [[TMP2]], ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP2]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 [[SUB]], ptr [[ATOMIC_TEMP2]], align 4
// CHECK32-NEXT:    [[CALL:%.*]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP1]], ptr noundef [[ATOMIC_TEMP2]], i32 noundef 5, i32 noundef 5)
// CHECK32-NEXT:    [[TMP3]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    br i1 [[CALL]], label [[ATOMIC_CONT:%.*]], label [[ATOMIC_OP]]
// CHECK32:       atomic_cont:
// CHECK32-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP4]], ptr noundef [[ATOMIC_TEMP3]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP5:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP3]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP5]]
//
long double test_volatile_compassign(volatile _Atomic long double *addr) {
  *addr -= 25;
  return *addr;
}

// CHECK-LABEL: define dso_local x86_fp80 @test_volatile_assign(
// CHECK-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 16
// CHECK-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 16 [[ATOMIC_TEMP]], i8 0, i64 16, i1 false)
// CHECK-NEXT:    store x86_fp80 0xK4005E600000000000000, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load i128, ptr [[ATOMIC_TEMP]], align 16
// CHECK-NEXT:    store atomic volatile i128 [[TMP1]], ptr [[TMP0]] seq_cst, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 8
// CHECK-NEXT:    [[ATOMIC_LOAD:%.*]] = load atomic volatile i128, ptr [[TMP2]] seq_cst, align 16
// CHECK-NEXT:    store i128 [[ATOMIC_LOAD]], ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 16
// CHECK-NEXT:    ret x86_fp80 [[TMP3]]
//
// CHECK32-LABEL: define dso_local x86_fp80 @test_volatile_assign(
// CHECK32-SAME: ptr noundef [[ADDR:%.*]]) #[[ATTR0]] {
// CHECK32-NEXT:  entry:
// CHECK32-NEXT:    [[ADDR_ADDR:%.*]] = alloca ptr, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    [[ATOMIC_TEMP1:%.*]] = alloca x86_fp80, align 4
// CHECK32-NEXT:    store ptr [[ADDR]], ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[ATOMIC_TEMP]], i8 0, i64 12, i1 false)
// CHECK32-NEXT:    store x86_fp80 0xK4005E600000000000000, ptr [[ATOMIC_TEMP]], align 4
// CHECK32-NEXT:    call void @__atomic_store(i32 noundef 12, ptr noundef [[TMP0]], ptr noundef [[ATOMIC_TEMP]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[ADDR_ADDR]], align 4
// CHECK32-NEXT:    call void @__atomic_load(i32 noundef 12, ptr noundef [[TMP1]], ptr noundef [[ATOMIC_TEMP1]], i32 noundef 5)
// CHECK32-NEXT:    [[TMP2:%.*]] = load x86_fp80, ptr [[ATOMIC_TEMP1]], align 4
// CHECK32-NEXT:    ret x86_fp80 [[TMP2]]
//
long double test_volatile_assign(volatile _Atomic long double *addr) {
  *addr = 115;

  return *addr;
}
