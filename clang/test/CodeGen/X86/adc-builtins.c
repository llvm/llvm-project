// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

#include <x86intrin.h>

unsigned char test_addcarry_u32(unsigned char __cf, unsigned int __x,
                                unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_addcarry_u32
// CHECK: [[ADC:%.*]] = call { i8, i32 } @llvm.x86.addcarry.32
// CHECK: [[DATA:%.*]] = extractvalue { i8, i32 } [[ADC]], 1
// CHECK: store i32 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i32 } [[ADC]], 0
  return _addcarry_u32(__cf, __x, __y, __p);
}

unsigned char test_addcarry_u64(unsigned char __cf, unsigned long long __x,
                                unsigned long long __y,
                                unsigned long long *__p) {
// CHECK-LABEL: test_addcarry_u64
// CHECK: [[ADC:%.*]] = call { i8, i64 } @llvm.x86.addcarry.64
// CHECK: [[DATA:%.*]] = extractvalue { i8, i64 } [[ADC]], 1
// CHECK: store i64 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i64 } [[ADC]], 0
  return _addcarry_u64(__cf, __x, __y, __p);
}

unsigned char test_subborrow_u32(unsigned char __cf, unsigned int __x,
                                 unsigned int __y, unsigned int *__p) {
// CHECK-LABEL: test_subborrow_u32
// CHECK: [[SBB:%.*]] = call { i8, i32 } @llvm.x86.subborrow.32
// CHECK: [[DATA:%.*]] = extractvalue { i8, i32 } [[SBB]], 1
// CHECK: store i32 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i32 } [[SBB]], 0
  return _subborrow_u32(__cf, __x, __y, __p);
}

unsigned char test_subborrow_u64(unsigned char __cf, unsigned long long __x,
                                 unsigned long long __y,
                                 unsigned long long *__p) {
// CHECK-LABEL: test_subborrow_u64
// CHECK: [[SBB:%.*]] = call { i8, i64 } @llvm.x86.subborrow.64
// CHECK: [[DATA:%.*]] = extractvalue { i8, i64 } [[SBB]], 1
// CHECK: store i64 [[DATA]], ptr %{{.*}}
// CHECK: [[CF:%.*]] = extractvalue { i8, i64 } [[SBB]], 0
  return _subborrow_u64(__cf, __x, __y, __p);
}

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)

template<typename X>
struct Result {
  unsigned char A;
  X B;
  constexpr bool operator==(const Result<X> &Other) {
    return A == Other.A && B == Other.B;
  }
};

constexpr Result<unsigned int>
const_test_addcarry_u32(unsigned char __cf, unsigned int __x, unsigned int __y)
{
  unsigned int __r{};
  return { _addcarry_u32(__cf, __x, __y, &__r), __r };
}

void constexpr adcu32() {
  static_assert(const_test_addcarry_u32(0, 0x00000000, 0x00000000) == Result<unsigned int>{0, 0x00000000});
  static_assert(const_test_addcarry_u32(1, 0xFFFFFFFE, 0x00000000) == Result<unsigned int>{0, 0xFFFFFFFF});
  static_assert(const_test_addcarry_u32(1, 0xFFFFFFFE, 0x00000001) == Result<unsigned int>{1, 0x00000000});
  static_assert(const_test_addcarry_u32(0, 0xFFFFFFFF, 0xFFFFFFFF) == Result<unsigned int>{1, 0xFFFFFFFE});
  static_assert(const_test_addcarry_u32(1, 0xFFFFFFFF, 0xFFFFFFFF) == Result<unsigned int>{1, 0xFFFFFFFF});
}

constexpr Result<unsigned int>
const_test_subborrow_u32(unsigned char __cf, unsigned int __x, unsigned int __y)
{
  unsigned int __r{};
  return { _subborrow_u32(__cf, __x, __y, &__r), __r };
}

void constexpr sbbu32() {
  static_assert(const_test_subborrow_u32(0, 0x00000000, 0x00000000) == Result<unsigned int>{0, 0x00000000});
  static_assert(const_test_subborrow_u32(0, 0x00000000, 0x00000001) == Result<unsigned int>{1, 0xFFFFFFFF});
  static_assert(const_test_subborrow_u32(1, 0x00000000, 0x00000001) == Result<unsigned int>{1, 0xFFFFFFFE});
  static_assert(const_test_subborrow_u32(1, 0xFFFFFFFE, 0x00000000) == Result<unsigned int>{0, 0xFFFFFFFD});
  static_assert(const_test_subborrow_u32(1, 0xFFFFFFFE, 0x00000001) == Result<unsigned int>{0, 0xFFFFFFFC});
  static_assert(const_test_subborrow_u32(0, 0xFFFFFFFF, 0xFFFFFFFF) == Result<unsigned int>{0, 0x00000000});
  static_assert(const_test_subborrow_u32(1, 0xFFFFFFFF, 0xFFFFFFFF) == Result<unsigned int>{1, 0xFFFFFFFF});
}

constexpr Result<unsigned long long>
const_test_addcarry_u64(unsigned char __cf, unsigned long long __x, unsigned long long __y)
{
  unsigned long long __r{};
  return { _addcarry_u64(__cf, __x, __y, &__r), __r };
}

void constexpr adcu64() {
  static_assert(const_test_addcarry_u64(0, 0x0000000000000000ULL, 0x0000000000000000ULL) == Result<unsigned long long>{0, 0x0000000000000000ULL});
  static_assert(const_test_addcarry_u64(1, 0xFFFFFFFFFFFFFFFEULL, 0x0000000000000000ULL) == Result<unsigned long long>{0, 0xFFFFFFFFFFFFFFFFULL});
  static_assert(const_test_addcarry_u64(1, 0xFFFFFFFFFFFFFFFEULL, 0x0000000000000001ULL) == Result<unsigned long long>{1, 0x0000000000000000ULL});
  static_assert(const_test_addcarry_u64(0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL) == Result<unsigned long long>{1, 0xFFFFFFFFFFFFFFFEULL});
  static_assert(const_test_addcarry_u64(1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL) == Result<unsigned long long>{1, 0xFFFFFFFFFFFFFFFFULL});
}

constexpr Result<unsigned long long>
const_test_subborrow_u64(unsigned char __cf, unsigned long long __x, unsigned long long __y)
{
  unsigned long long __r{};
  return { _subborrow_u64(__cf, __x, __y, &__r), __r };
}

void constexpr sbbu64() {
  static_assert(const_test_subborrow_u64(0, 0x0000000000000000ULL, 0x0000000000000000ULL) == Result<unsigned long long>{0, 0x0000000000000000ULL});
  static_assert(const_test_subborrow_u64(0, 0x0000000000000000ULL, 0x0000000000000001ULL) == Result<unsigned long long>{1, 0xFFFFFFFFFFFFFFFFULL});
  static_assert(const_test_subborrow_u64(1, 0x0000000000000000ULL, 0x0000000000000001ULL) == Result<unsigned long long>{1, 0xFFFFFFFFFFFFFFFEULL});
  static_assert(const_test_subborrow_u64(1, 0xFFFFFFFFFFFFFFFEULL, 0x0000000000000000ULL) == Result<unsigned long long>{0, 0xFFFFFFFFFFFFFFFDULL});
  static_assert(const_test_subborrow_u64(1, 0xFFFFFFFFFFFFFFFEULL, 0x0000000000000001ULL) == Result<unsigned long long>{0, 0xFFFFFFFFFFFFFFFCULL});
  static_assert(const_test_subborrow_u64(0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL) == Result<unsigned long long>{0, 0x0000000000000000ULL});
  static_assert(const_test_subborrow_u64(1, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL) == Result<unsigned long long>{1, 0xFFFFFFFFFFFFFFFFULL});
}

#endif