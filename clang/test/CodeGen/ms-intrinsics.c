// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s -check-prefixes CHECK,CHECK-I386,CHECK-INTEL
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple thumbv7--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-ARM,CHECK-ARM-X64
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -Oz -emit-llvm -target-feature +cx16 %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-X64,CHECK-ARM-X64,CHECK-INTEL

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

#if defined(__i386__) || defined(__x86_64__)
void test__stosb(unsigned char *Dest, unsigned char Data, size_t Count) {
  return __stosb(Dest, Data, Count);
}

// CHECK-I386: define{{.*}}void @test__stosb
// CHECK-I386:   tail call void @llvm.memset.p0i8.i32(i8* align 1 %Dest, i8 %Data, i32 %Count, i1 true)
// CHECK-I386:   ret void
// CHECK-I386: }

// CHECK-X64: define{{.*}}void @test__stosb
// CHECK-X64:   tail call void @llvm.memset.p0i8.i64(i8* align 1 %Dest, i8 %Data, i64 %Count, i1 true)
// CHECK-X64:   ret void
// CHECK-X64: }

void test__ud2(void) {
  __ud2();
}
// CHECK-INTEL-LABEL: define{{.*}} void @test__ud2()
// CHECK-INTEL: call void @llvm.trap()

void test__int2c(void) {
  __int2c();
}
// CHECK-INTEL-LABEL: define{{.*}} void @test__int2c()
// CHECK-INTEL: call void asm sideeffect "int $$0x2c", ""() #[[NORETURN:[0-9]+]]


#endif

void *test_ReturnAddress() {
  return _ReturnAddress();
}
// CHECK-LABEL: define{{.*}}i8* @test_ReturnAddress()
// CHECK: = tail call i8* @llvm.returnaddress(i32 0)
// CHECK: ret i8*

#if defined(__i386__) || defined(__x86_64__)
void *test_AddressOfReturnAddress() {
  return _AddressOfReturnAddress();
}
// CHECK-INTEL-LABEL: define dso_local i8* @test_AddressOfReturnAddress()
// CHECK-INTEL: = tail call i8* @llvm.addressofreturnaddress()
// CHECK-INTEL: ret i8*
#endif

unsigned char test_BitScanForward(unsigned long *Index, unsigned long Mask) {
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

unsigned char test_BitScanReverse(unsigned long *Index, unsigned long Mask) {
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

#if defined(__x86_64__) || defined(__arm__)
unsigned char test_BitScanForward64(unsigned long *Index, unsigned __int64 Mask) {
  return _BitScanForward64(Index, Mask);
}
// CHECK-ARM-X64: define{{.*}}i8 @test_BitScanForward64(i32* {{[a-z_ ]*}}%Index, i64 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK-ARM-X64:   [[ISNOTZERO:%[a-z0-9._]+]] = icmp eq i64 %Mask, 0
// CHECK-ARM-X64:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK-ARM-X64:   [[END_LABEL]]:
// CHECK-ARM-X64:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK-ARM-X64:   ret i8 [[RESULT]]
// CHECK-ARM-X64:   [[ISNOTZERO_LABEL]]:
// CHECK-ARM-X64:   [[INDEX:%[0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %Mask, i1 true)
// CHECK-ARM-X64:   [[TRUNC_INDEX:%[0-9]+]] = trunc i64 [[INDEX]] to i32
// CHECK-ARM-X64:   store i32 [[TRUNC_INDEX]], i32* %Index, align 4
// CHECK-ARM-X64:   br label %[[END_LABEL]]

unsigned char test_BitScanReverse64(unsigned long *Index, unsigned __int64 Mask) {
  return _BitScanReverse64(Index, Mask);
}
// CHECK-ARM-X64: define{{.*}}i8 @test_BitScanReverse64(i32* {{[a-z_ ]*}}%Index, i64 {{[a-z_ ]*}}%Mask){{.*}}{
// CHECK-ARM-X64:   [[ISNOTZERO:%[0-9]+]] = icmp eq i64 %Mask, 0
// CHECK-ARM-X64:   br i1 [[ISNOTZERO]], label %[[END_LABEL:[a-z0-9._]+]], label %[[ISNOTZERO_LABEL:[a-z0-9._]+]]
// CHECK-ARM-X64:   [[END_LABEL]]:
// CHECK-ARM-X64:   [[RESULT:%[a-z0-9._]+]] = phi i8 [ 0, %[[ISZERO_LABEL:[a-z0-9._]+]] ], [ 1, %[[ISNOTZERO_LABEL]] ]
// CHECK-ARM-X64:   ret i8 [[RESULT]]
// CHECK-ARM-X64:   [[ISNOTZERO_LABEL]]:
// CHECK-ARM-X64:   [[REVINDEX:%[0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %Mask, i1 true)
// CHECK-ARM-X64:   [[TRUNC_REVINDEX:%[0-9]+]] = trunc i64 [[REVINDEX]] to i32
// CHECK-ARM-X64:   [[INDEX:%[0-9]+]] = xor i32 [[TRUNC_REVINDEX]], 63
// CHECK-ARM-X64:   store i32 [[INDEX]], i32* %Index, align 4
// CHECK-ARM-X64:   br label %[[END_LABEL]]
#endif

void *test_InterlockedExchangePointer(void * volatile *Target, void *Value) {
  return _InterlockedExchangePointer(Target, Value);
}

// CHECK: define{{.*}}i8* @test_InterlockedExchangePointer(i8** {{[a-z_ ]*}}%Target, i8* {{[a-z_ ]*}}%Value){{.*}}{
// CHECK:   %[[TARGET:[0-9]+]] = bitcast i8** %Target to [[iPTR:i[0-9]+]]*
// CHECK:   %[[VALUE:[0-9]+]] = ptrtoint i8* %Value to [[iPTR]]
// CHECK:   %[[EXCHANGE:[0-9]+]] = atomicrmw xchg [[iPTR]]* %[[TARGET]], [[iPTR]] %[[VALUE]] seq_cst
// CHECK:   %[[RESULT:[0-9]+]] = inttoptr [[iPTR]] %[[EXCHANGE]] to i8*
// CHECK:   ret i8* %[[RESULT]]
// CHECK: }

void *test_InterlockedCompareExchangePointer(void * volatile *Destination,
                                             void *Exchange, void *Comparand) {
  return _InterlockedCompareExchangePointer(Destination, Exchange, Comparand);
}

// CHECK: define{{.*}}i8* @test_InterlockedCompareExchangePointer(i8** {{[a-z_ ]*}}%Destination, i8* {{[a-z_ ]*}}%Exchange, i8* {{[a-z_ ]*}}%Comparand){{.*}}{
// CHECK:   %[[DEST:[0-9]+]] = bitcast i8** %Destination to [[iPTR]]*
// CHECK:   %[[EXCHANGE:[0-9]+]] = ptrtoint i8* %Exchange to [[iPTR]]
// CHECK:   %[[COMPARAND:[0-9]+]] = ptrtoint i8* %Comparand to [[iPTR]]
// CHECK:   %[[XCHG:[0-9]+]] = cmpxchg volatile [[iPTR]]* %[[DEST:[0-9]+]], [[iPTR]] %[[COMPARAND:[0-9]+]], [[iPTR]] %[[EXCHANGE:[0-9]+]] seq_cst seq_cst
// CHECK:   %[[EXTRACT:[0-9]+]] = extractvalue { [[iPTR]], i1 } %[[XCHG]], 0
// CHECK:   %[[RESULT:[0-9]+]] = inttoptr [[iPTR]] %[[EXTRACT]] to i8*
// CHECK:   ret i8* %[[RESULT:[0-9]+]]
// CHECK: }

char test_InterlockedExchange8(char volatile *value, char mask) {
  return _InterlockedExchange8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedExchange8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xchg i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedExchange16(short volatile *value, short mask) {
  return _InterlockedExchange16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedExchange16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xchg i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedExchange(long volatile *value, long mask) {
  return _InterlockedExchange(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchange(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xchg i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedExchangeAdd8(char volatile *value, char mask) {
  return _InterlockedExchangeAdd8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedExchangeAdd8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw add i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedExchangeAdd16(short volatile *value, short mask) {
  return _InterlockedExchangeAdd16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedExchangeAdd16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw add i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedExchangeAdd(long volatile *value, long mask) {
  return _InterlockedExchangeAdd(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchangeAdd(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw add i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedExchangeSub8(char volatile *value, char mask) {
  return _InterlockedExchangeSub8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedExchangeSub8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw sub i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedExchangeSub16(short volatile *value, short mask) {
  return _InterlockedExchangeSub16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedExchangeSub16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw sub i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedExchangeSub(long volatile *value, long mask) {
  return _InterlockedExchangeSub(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedExchangeSub(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw sub i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedOr8(char volatile *value, char mask) {
  return _InterlockedOr8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedOr8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw or i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedOr16(short volatile *value, short mask) {
  return _InterlockedOr16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedOr16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw or i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedOr(long volatile *value, long mask) {
  return _InterlockedOr(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedOr(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw or i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedXor8(char volatile *value, char mask) {
  return _InterlockedXor8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedXor8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xor i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedXor16(short volatile *value, short mask) {
  return _InterlockedXor16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedXor16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xor i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedXor(long volatile *value, long mask) {
  return _InterlockedXor(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedXor(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xor i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedAnd8(char volatile *value, char mask) {
  return _InterlockedAnd8(value, mask);
}
// CHECK: define{{.*}}i8 @test_InterlockedAnd8(i8*{{[a-z_ ]*}}%value, i8{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw and i8* %value, i8 %mask seq_cst
// CHECK:   ret i8 [[RESULT:%[0-9]+]]
// CHECK: }

short test_InterlockedAnd16(short volatile *value, short mask) {
  return _InterlockedAnd16(value, mask);
}
// CHECK: define{{.*}}i16 @test_InterlockedAnd16(i16*{{[a-z_ ]*}}%value, i16{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw and i16* %value, i16 %mask seq_cst
// CHECK:   ret i16 [[RESULT:%[0-9]+]]
// CHECK: }

long test_InterlockedAnd(long volatile *value, long mask) {
  return _InterlockedAnd(value, mask);
}
// CHECK: define{{.*}}i32 @test_InterlockedAnd(i32*{{[a-z_ ]*}}%value, i32{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw and i32* %value, i32 %mask seq_cst
// CHECK:   ret i32 [[RESULT:%[0-9]+]]
// CHECK: }

char test_InterlockedCompareExchange8(char volatile *Destination, char Exchange, char Comperand) {
  return _InterlockedCompareExchange8(Destination, Exchange, Comperand);
}
// CHECK: define{{.*}}i8 @test_InterlockedCompareExchange8(i8*{{[a-z_ ]*}}%Destination, i8{{[a-z_ ]*}}%Exchange, i8{{[a-z_ ]*}}%Comperand){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = cmpxchg volatile i8* %Destination, i8 %Comperand, i8 %Exchange seq_cst seq_cst
// CHECK: [[RESULT:%[0-9]+]] = extractvalue { i8, i1 } [[TMP]], 0
// CHECK: ret i8 [[RESULT]]
// CHECK: }

short test_InterlockedCompareExchange16(short volatile *Destination, short Exchange, short Comperand) {
  return _InterlockedCompareExchange16(Destination, Exchange, Comperand);
}
// CHECK: define{{.*}}i16 @test_InterlockedCompareExchange16(i16*{{[a-z_ ]*}}%Destination, i16{{[a-z_ ]*}}%Exchange, i16{{[a-z_ ]*}}%Comperand){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = cmpxchg volatile i16* %Destination, i16 %Comperand, i16 %Exchange seq_cst seq_cst
// CHECK: [[RESULT:%[0-9]+]] = extractvalue { i16, i1 } [[TMP]], 0
// CHECK: ret i16 [[RESULT]]
// CHECK: }

long test_InterlockedCompareExchange(long volatile *Destination, long Exchange, long Comperand) {
  return _InterlockedCompareExchange(Destination, Exchange, Comperand);
}
// CHECK: define{{.*}}i32 @test_InterlockedCompareExchange(i32*{{[a-z_ ]*}}%Destination, i32{{[a-z_ ]*}}%Exchange, i32{{[a-z_ ]*}}%Comperand){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = cmpxchg volatile i32* %Destination, i32 %Comperand, i32 %Exchange seq_cst seq_cst
// CHECK: [[RESULT:%[0-9]+]] = extractvalue { i32, i1 } [[TMP]], 0
// CHECK: ret i32 [[RESULT]]
// CHECK: }

__int64 test_InterlockedCompareExchange64(__int64 volatile *Destination, __int64 Exchange, __int64 Comperand) {
  return _InterlockedCompareExchange64(Destination, Exchange, Comperand);
}
// CHECK: define{{.*}}i64 @test_InterlockedCompareExchange64(i64*{{[a-z_ ]*}}%Destination, i64{{[a-z_ ]*}}%Exchange, i64{{[a-z_ ]*}}%Comperand){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = cmpxchg volatile i64* %Destination, i64 %Comperand, i64 %Exchange seq_cst seq_cst
// CHECK: [[RESULT:%[0-9]+]] = extractvalue { i64, i1 } [[TMP]], 0
// CHECK: ret i64 [[RESULT]]
// CHECK: }

#if defined(__x86_64__)
unsigned char test_InterlockedCompareExchange128(__int64 volatile *Destination, __int64 ExchangeHigh, __int64 ExchangeLow, __int64* ComparandResult) {
  return _InterlockedCompareExchange128(Destination, ExchangeHigh, ExchangeLow, ComparandResult);
}
// CHECK-X64: define{{.*}}i8 @test_InterlockedCompareExchange128(i64*{{[a-z_ ]*}}%Destination, i64{{[a-z_ ]*}}%ExchangeHigh, i64{{[a-z_ ]*}}%ExchangeLow, i64*{{[a-z_ ]*}}%ComparandResult){{.*}}{
// CHECK-X64: [[DST:%[0-9]+]] = bitcast i64* %Destination to i128*
// CHECK-X64: [[EH:%[0-9]+]] = zext i64 %ExchangeHigh to i128
// CHECK-X64: [[EL:%[0-9]+]] = zext i64 %ExchangeLow to i128
// CHECK-X64: [[CNR:%[0-9]+]] = bitcast i64* %ComparandResult to i128*
// CHECK-X64: [[EHS:%[0-9]+]] = shl nuw i128 [[EH]], 64
// CHECK-X64: [[EXP:%[0-9]+]] = or i128 [[EHS]], [[EL]]
// CHECK-X64: [[ORG:%[0-9]+]] = load i128, i128* [[CNR]], align 16
// CHECK-X64: [[RES:%[0-9]+]] = cmpxchg volatile i128* [[DST]], i128 [[ORG]], i128 [[EXP]] seq_cst seq_cst
// CHECK-X64: [[OLD:%[0-9]+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK-X64: store i128 [[OLD]], i128* [[CNR]], align 16
// CHECK-X64: [[SUC1:%[0-9]+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK-X64: [[SUC8:%[0-9]+]] = zext i1 [[SUC1]] to i8
// CHECK-X64: ret i8 [[SUC8]]
// CHECK-X64: }
#endif

short test_InterlockedIncrement16(short volatile *Addend) {
  return _InterlockedIncrement16(Addend);
}
// CHECK: define{{.*}}i16 @test_InterlockedIncrement16(i16*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw add i16* %Addend, i16 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i16 [[TMP]], 1
// CHECK: ret i16 [[RESULT]]
// CHECK: }

long test_InterlockedIncrement(long volatile *Addend) {
  return _InterlockedIncrement(Addend);
}
// CHECK: define{{.*}}i32 @test_InterlockedIncrement(i32*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw add i32* %Addend, i32 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i32 [[TMP]], 1
// CHECK: ret i32 [[RESULT]]
// CHECK: }

short test_InterlockedDecrement16(short volatile *Addend) {
  return _InterlockedDecrement16(Addend);
}
// CHECK: define{{.*}}i16 @test_InterlockedDecrement16(i16*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw sub i16* %Addend, i16 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i16 [[TMP]], -1
// CHECK: ret i16 [[RESULT]]
// CHECK: }

long test_InterlockedDecrement(long volatile *Addend) {
  return _InterlockedDecrement(Addend);
}
// CHECK: define{{.*}}i32 @test_InterlockedDecrement(i32*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw sub i32* %Addend, i32 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i32 [[TMP]], -1
// CHECK: ret i32 [[RESULT]]
// CHECK: }

#if defined(__x86_64__) || defined(__arm__)
__int64 test_InterlockedExchange64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchange64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedExchange64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw xchg i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedExchangeAdd64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchangeAdd64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedExchangeAdd64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw add i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedExchangeSub64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchangeSub64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedExchangeSub64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw sub i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedOr64(__int64 volatile *value, __int64 mask) {
  return _InterlockedOr64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedOr64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw or i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedXor64(__int64 volatile *value, __int64 mask) {
  return _InterlockedXor64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedXor64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw xor i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedAnd64(__int64 volatile *value, __int64 mask) {
  return _InterlockedAnd64(value, mask);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedAnd64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK-ARM-X64:   [[RESULT:%[0-9]+]] = atomicrmw and i64* %value, i64 %mask seq_cst
// CHECK-ARM-X64:   ret i64 [[RESULT:%[0-9]+]]
// CHECK-ARM-X64: }

__int64 test_InterlockedIncrement64(__int64 volatile *Addend) {
  return _InterlockedIncrement64(Addend);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedIncrement64(i64*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK-ARM-X64: [[TMP:%[0-9]+]] = atomicrmw add i64* %Addend, i64 1 seq_cst
// CHECK-ARM-X64: [[RESULT:%[0-9]+]] = add i64 [[TMP]], 1
// CHECK-ARM-X64: ret i64 [[RESULT]]
// CHECK-ARM-X64: }

__int64 test_InterlockedDecrement64(__int64 volatile *Addend) {
  return _InterlockedDecrement64(Addend);
}
// CHECK-ARM-X64: define{{.*}}i64 @test_InterlockedDecrement64(i64*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK-ARM-X64: [[TMP:%[0-9]+]] = atomicrmw sub i64* %Addend, i64 1 seq_cst
// CHECK-ARM-X64: [[RESULT:%[0-9]+]] = add i64 [[TMP]], -1
// CHECK-ARM-X64: ret i64 [[RESULT]]
// CHECK-ARM-X64: }

#endif

void test__fastfail() {
  __fastfail(42);
}
// CHECK-LABEL: define{{.*}} void @test__fastfail()
// CHECK-ARM: call void asm sideeffect "udf #251", "{r0}"(i32 42) #[[NORETURN:[0-9]+]]
// CHECK-INTEL: call void asm sideeffect "int $$0x29", "{cx}"(i32 42) #[[NORETURN]]

// Attributes come last.

// CHECK: attributes #[[NORETURN]] = { noreturn{{.*}} }

