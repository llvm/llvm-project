// RUN: %clang_cc1 -x c -ffreestanding -triple i686--linux -no-enable-noundef-analysis -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64--linux -no-enable-noundef-analysis -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

// RUN: %clang_cc1 -x c++ -ffreestanding -triple i686--linux -no-enable-noundef-analysis -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64--linux -no-enable-noundef-analysis -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

// RUN: %clang_cc1 -x c++ -ffreestanding -triple i686--linux -no-enable-noundef-analysis -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64--linux -no-enable-noundef-analysis -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -no-enable-noundef-analysis -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

#include <x86intrin.h>
#include "builtin_test_helpers.h"

unsigned char test__rolb(unsigned char value, int shift) {
// CHECK-LABEL: test__rolb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rolb(value, shift);
}
TEST_CONSTEXPR(__rolb(0x01, 5) == 0x20);

unsigned short test__rolw(unsigned short value, int shift) {
// CHECK-LABEL: test__rolw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rolw(value, shift);
}
TEST_CONSTEXPR(__rolw(0x3210, 11) == 0x8190);

unsigned int test__rold(unsigned int value, int shift) {
// CHECK-LABEL: test__rold
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rold(value, shift);
}
TEST_CONSTEXPR(__rold(0x76543210, 22) == 0x841D950C);

#if defined(__x86_64__)
unsigned long test__rolq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: test__rolq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rolq(value, shift);
}
TEST_CONSTEXPR(__rolq(0xFEDCBA9876543210ULL, 55) == 0x087F6E5D4C3B2A19ULL);
#endif

unsigned char test__rorb(unsigned char value, int shift) {
// CHECK-LABEL: test__rorb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rorb(value, shift);
}
TEST_CONSTEXPR(__rorb(0x01, 5) == 0x08);

unsigned short test__rorw(unsigned short value, int shift) {
// CHECK-LABEL: test__rorw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rorw(value, shift);
}
TEST_CONSTEXPR(__rorw(0x3210, 11) == 0x4206);

unsigned int test__rord(unsigned int value, int shift) {
// CHECK-LABEL: test__rord
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rord(value, shift);
}
TEST_CONSTEXPR(__rord(0x76543210, 22) == 0x50C841D9);

#if defined(__x86_64__)
unsigned long test__rorq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: test__rorq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rorq(value, shift);
}
TEST_CONSTEXPR(__rorq(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL);
#endif

unsigned short test_rotwl(unsigned short value, int shift) {
// CHECK-LABEL: test_rotwl
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwl(value, shift);
}
TEST_CONSTEXPR(_rotwl(0x3210, 4) == 0x2103);

unsigned int test_rotl(unsigned int value, int shift) {
// CHECK-LABEL: test_rotl
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotl(value, shift);
}
TEST_CONSTEXPR(_rotl(0x76543210, 8) == 0x54321076);

unsigned long test_lrotl(unsigned long value, int shift) {
// CHECK-32BIT-LONG-LABEL: test_lrotl
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
//
// CHECK-64BIT-LONG-LABEL: test_lrotl
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]
  return _lrotl(value, shift);
}
#if defined(__LP64__) && !defined(_MSC_VER)
TEST_CONSTEXPR(_lrotl(0xFEDCBA9876543210ULL, 55) == 0x087F6E5D4C3B2A19ULL);
#else
TEST_CONSTEXPR(_lrotl(0x76543210, 22) == 0x841D950C);
#endif


unsigned short test_rotwr(unsigned short value, int shift) {
// CHECK-LABEL: test_rotwr
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwr(value, shift);
}
TEST_CONSTEXPR(_rotwr(0x3210, 4) == 0x0321);

unsigned int test_rotr(unsigned int value, int shift) {
// CHECK-LABEL: test_rotr
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotr(value, shift);
}
TEST_CONSTEXPR(_rotr(0x76543210, 8) == 0x10765432);

unsigned long test_lrotr(unsigned long value, int shift) {
// CHECK-32BIT-LONG-LABEL: test_lrotr
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
//
// CHECK-64BIT-LONG-LABEL: test_lrotr
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]
  return _lrotr(value, shift);
}
#if defined(__LP64__) && !defined(_MSC_VER)
TEST_CONSTEXPR(_lrotr(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL);
#else
TEST_CONSTEXPR(_lrotr(0x76543210, 22) == 0x50C841D9);
#endif

