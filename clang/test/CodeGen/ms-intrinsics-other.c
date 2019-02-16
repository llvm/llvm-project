// RUN: %clang_cc1 -ffreestanding -fms-extensions \
// RUN:         -triple x86_64--darwin -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -fms-extensions \
// RUN:         -triple x86_64--linux -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s

// LP64 targets use 'long' as 'int' for MS intrinsics (-fms-extensions)
#ifdef __LP64__
#define LONG int
#else
#define LONG long
#endif

unsigned char test_BitScanForward(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanForward(Index, Mask);
}
// CHECK: define{{.*}}i8 @test_BitScanForward(i32* {{[a-z_ ]*}}%Index, i32 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK:   [[ISNOTZERO:%[a-z0-9._]+]] = icmp eq i32 %Mask, 0
// CHECK:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK:   [[END_LABEL]]:
// CHECK:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK:   ret i8 [[RESULT]]
// CHECK:   [[ISNOTZERO_LABEL]]:
// CHECK:   [[INDEX:%[0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %Mask, i1 true)
// CHECK:   store i32 [[INDEX]], i32* %Index, align 4
// CHECK:   br label %[[END_LABEL]]

unsigned char test_BitScanReverse(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanReverse(Index, Mask);
}
// CHECK: define{{.*}}i8 @test_BitScanReverse(i32* {{[a-z_ ]*}}%Index, i32 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK:   [[ISNOTZERO:%[0-9]+]] = icmp eq i32 %Mask, 0
// CHECK:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK:   [[END_LABEL]]:
// CHECK:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK:   ret i8 [[RESULT]]
// CHECK:   [[ISNOTZERO_LABEL]]:
// CHECK:   [[REVINDEX:%[0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %Mask, i1 true)
// CHECK:   [[INDEX:%[0-9]+]] = xor i32 [[REVINDEX]], 31
// CHECK:   store i32 [[INDEX]], i32* %Index, align 4
// CHECK:   br label %[[END_LABEL]]

#if defined(__x86_64__)
unsigned char test_BitScanForward64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanForward64(Index, Mask);
}
// CHECK: define{{.*}}i8 @test_BitScanForward64(i32* {{[a-z_ ]*}}%Index, i64 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK:   [[ISNOTZERO:%[a-z0-9._]+]] = icmp eq i64 %Mask, 0
// CHECK:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK:   [[END_LABEL]]:
// CHECK:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK:   ret i8 [[RESULT]]
// CHECK:   [[ISNOTZERO_LABEL]]:
// CHECK:   [[INDEX:%[0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %Mask, i1 true)
// CHECK:   [[TRUNC_INDEX:%[0-9]+]] = trunc i64 [[INDEX]] to i32
// CHECK:   store i32 [[TRUNC_INDEX]], i32* %Index, align 4
// CHECK:   br label %[[END_LABEL]]

unsigned char test_BitScanReverse64(unsigned LONG *Index, unsigned __int64 Mask) {
  return _BitScanReverse64(Index, Mask);
}
// CHECK: define{{.*}}i8 @test_BitScanReverse64(i32* {{[a-z_ ]*}}%Index, i64 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK:   [[ISNOTZERO:%[0-9]+]] = icmp eq i64 %Mask, 0
// CHECK:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK:   [[END_LABEL]]:
// CHECK:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK:   ret i8 [[RESULT]]
// CHECK:   [[ISNOTZERO_LABEL]]:
// CHECK:   [[REVINDEX:%[0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %Mask, i1 true)
// CHECK:   [[TRUNC_REVINDEX:%[0-9]+]] = trunc i64 [[REVINDEX]] to i32
// CHECK:   [[INDEX:%[0-9]+]] = xor i32 [[TRUNC_REVINDEX]], 63
// CHECK:   store i32 [[INDEX]], i32* %Index, align 4
// CHECK:   br label %[[END_LABEL]]
#endif

LONG test_InterlockedExchange(LONG volatile *value, LONG mask) {
  return _InterlockedExchange(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchange(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xchg i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedExchangeAdd(LONG volatile *value, LONG mask) {
  return _InterlockedExchangeAdd(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchangeAdd(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw add i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedExchangeSub(LONG volatile *value, LONG mask) {
  return _InterlockedExchangeSub(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchangeSub(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw sub i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedOr(LONG volatile *value, LONG mask) {
  return _InterlockedOr(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedOr(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw or i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedXor(LONG volatile *value, LONG mask) {
  return _InterlockedXor(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedXor(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xor i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedAnd(LONG volatile *value, LONG mask) {
  return _InterlockedAnd(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedAnd(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw and i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

LONG test_InterlockedCompareExchange(LONG volatile *Destination, LONG Exchange, LONG Comperand) {
  return _InterlockedCompareExchange(Destination, Exchange, Comperand);
}
// CHECK: define{{.*}}i32 @test_InterlockedCompareExchange(i32*{{[a-z_ ]*}}%Destination, i32{{[a-z_ ]*}}%Exchange, i32{{[a-z_ ]*}}%Comperand){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = cmpxchg volatile i32* %Destination, i32 %Comperand, i32 %Exchange seq_cst seq_cst
// CHECK: [[RESULT:%[0-9]+]] = extractvalue { i32, i1 } [[TMP]], 0
// CHECK: ret i32 [[RESULT]]
// CHECK: }

LONG test_InterlockedIncrement(LONG volatile *Addend) {
  return _InterlockedIncrement(Addend);
}
// CHECK: define{{.*}}i32 @test_InterlockedIncrement(i32*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw add i32* %Addend, i32 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i32 [[TMP]], 1
// CHECK: ret i32 [[RESULT]]
// CHECK: }

LONG test_InterlockedDecrement(LONG volatile *Addend) {
  return _InterlockedDecrement(Addend);
}
// CHECK: define{{.*}}i32 @test_InterlockedDecrement(i32*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw sub i32* %Addend, i32 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i32 [[TMP]], -1
// CHECK: ret i32 [[RESULT]]
// CHECK: }

unsigned short test__lzcnt16(unsigned short x) {
  return __lzcnt16(x);
}
// CHECK: i16 @test__lzcnt16
// CHECK:  [[RESULT:%[0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %x, i1 false)
// CHECK: ret i16 [[RESULT]]
// CHECK: }

unsigned int test__lzcnt(unsigned int x) {
  return __lzcnt(x);
}
// CHECK: i32 @test__lzcnt
// CHECK:  [[RESULT:%[0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
// CHECK: ret i32 [[RESULT]]
// CHECK: }

unsigned __int64 test__lzcnt64(unsigned __int64 x) {
  return __lzcnt64(x);
}
// CHECK: i64 @test__lzcnt64
// CHECK:  [[RESULT:%[0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %x, i1 false)
// CHECK: ret i64 [[RESULT]]
// CHECK: }

unsigned short test__popcnt16(unsigned short x) {
  return __popcnt16(x);
}
// CHECK: i16 @test__popcnt16
// CHECK:  [[RESULT:%[0-9]+]] = tail call i16 @llvm.ctpop.i16(i16 %x)
// CHECK: ret i16 [[RESULT]]
// CHECK: }

unsigned int test__popcnt(unsigned int x) {
  return __popcnt(x);
}
// CHECK: i32 @test__popcnt
// CHECK:  [[RESULT:%[0-9]+]] = tail call i32 @llvm.ctpop.i32(i32 %x)
// CHECK: ret i32 [[RESULT]]
// CHECK: }

unsigned __int64 test__popcnt64(unsigned __int64 x) {
  return __popcnt64(x);
}
// CHECK: i64 @test__popcnt64
// CHECK:  [[RESULT:%[0-9]+]] = tail call i64 @llvm.ctpop.i64(i64 %x)
// CHECK: ret i64 [[RESULT]]
// CHECK: }
