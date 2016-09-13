// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s -check-prefix CHECK -check-prefix CHECK-I386
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple thumbv7--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-X64

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

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

#if defined(__i386__)
long test__readfsdword(unsigned long Offset) {
  return __readfsdword(Offset);
}

// CHECK-I386: define i32 @test__readfsdword(i32 %Offset){{.*}}{
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %Offset to i32 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i32, i32 addrspace(257)* [[PTR]], align 4
// CHECK-I386:   ret i32 [[VALUE:%[0-9]+]]
// CHECK-I386: }
#endif

#if defined(__x86_64__)
unsigned __int64 test__umulh(unsigned __int64 a, unsigned __int64 b) {
  return __umulh(a, b);
}
// CHECK-X64-LABEL: define i64 @test__umulh(i64 %a, i64 %b)
// CHECK-X64: = mul nuw i128 %

#endif

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

__int64 test_InterlockedExchange64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchange64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedExchange64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xchg i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedExchangeAdd64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchangeAdd64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedExchangeAdd64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw add i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedExchangeSub64(__int64 volatile *value, __int64 mask) {
  return _InterlockedExchangeSub64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedExchangeSub64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw sub i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedOr64(__int64 volatile *value, __int64 mask) {
  return _InterlockedOr64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedOr64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw or i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedXor64(__int64 volatile *value, __int64 mask) {
  return _InterlockedXor64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedXor64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw xor i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedAnd64(__int64 volatile *value, __int64 mask) {
  return _InterlockedAnd64(value, mask);
}
// CHECK: define{{.*}}i64 @test_InterlockedAnd64(i64*{{[a-z_ ]*}}%value, i64{{[a-z_ ]*}}%mask){{.*}}{
// CHECK:   [[RESULT:%[0-9]+]] = atomicrmw and i64* %value, i64 %mask seq_cst
// CHECK:   ret i64 [[RESULT:%[0-9]+]]
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

__int64 test_InterlockedIncrement64(__int64 volatile *Addend) {
  return _InterlockedIncrement64(Addend);
}
// CHECK: define{{.*}}i64 @test_InterlockedIncrement64(i64*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw add i64* %Addend, i64 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i64 [[TMP]], 1
// CHECK: ret i64 [[RESULT]]
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

__int64 test_InterlockedDecrement64(__int64 volatile *Addend) {
  return _InterlockedDecrement64(Addend);
}
// CHECK: define{{.*}}i64 @test_InterlockedDecrement64(i64*{{[a-z_ ]*}}%Addend){{.*}}{
// CHECK: [[TMP:%[0-9]+]] = atomicrmw sub i64* %Addend, i64 1 seq_cst
// CHECK: [[RESULT:%[0-9]+]] = add i64 [[TMP]], -1
// CHECK: ret i64 [[RESULT]]
// CHECK: }
