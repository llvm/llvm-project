// RUN: %clang_cc1 -x c -ffreestanding -triple i686--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding -triple i686--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding -triple x86_64--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -x c++ -std=c++11 -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

#include <x86intrin.h>

unsigned char test__rolb(unsigned char value, int shift) {
// CHECK-LABEL: test__rolb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rolb(value, shift);
}

unsigned short test__rolw(unsigned short value, int shift) {
// CHECK-LABEL: test__rolw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rolw(value, shift);
}

unsigned int test__rold(unsigned int value, int shift) {
// CHECK-LABEL: test__rold
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rold(value, shift);
}

#if defined(__x86_64__)
unsigned long test__rolq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: test__rolq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rolq(value, shift);
}
#endif

unsigned char test__rorb(unsigned char value, int shift) {
// CHECK-LABEL: test__rorb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rorb(value, shift);
}

unsigned short test__rorw(unsigned short value, int shift) {
// CHECK-LABEL: test__rorw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rorw(value, shift);
}

unsigned int test__rord(unsigned int value, int shift) {
// CHECK-LABEL: test__rord
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rord(value, shift);
}

#if defined(__x86_64__)
unsigned long test__rorq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: test__rorq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rorq(value, shift);
}
#endif

unsigned short test_rotwl(unsigned short value, int shift) {
// CHECK-LABEL: test_rotwl
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwl(value, shift);
}

unsigned int test_rotl(unsigned int value, int shift) {
// CHECK-LABEL: test_rotl
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotl(value, shift);
}

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


unsigned short test_rotwr(unsigned short value, int shift) {
// CHECK-LABEL: test_rotwr
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwr(value, shift);
}

unsigned int test_rotr(unsigned int value, int shift) {
// CHECK-LABEL: test_rotr
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotr(value, shift);
}

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

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)

char rolb_0[__rolb(0x01, 5) == 0x20 ? 1 : -1];
char rolw_0[__rolw(0x3210, 11) == 0x8190 ? 1 : -1];
char rold_0[__rold(0x76543210, 22) == 0x841D950C ? 1 : -1];

char rorb_0[__rorb(0x01, 5) == 0x08 ? 1 : -1];
char rorw_0[__rorw(0x3210, 11) == 0x4206 ? 1 : -1];
char rord_0[__rord(0x76543210, 22) == 0x50C841D9 ? 1 : -1];

#if defined(__x86_64__)
char rolq_0[__rolq(0xFEDCBA9876543210ULL, 55) == 0x087F6E5D4C3B2A19ULL ? 1 : -1];
char rorq_0[__rorq(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL ? 1 : -1];
#endif

char rotwl_0[_rotwl(0x3210, 4) == 0x2103 ? 1 : -1];
char rotwr_0[_rotwr(0x3210, 4) == 0x0321 ? 1 : -1];
char rotl_0[_rotl(0x76543210, 8) == 0x54321076 ? 1 : -1];
char rotr_0[_rotr(0x76543210, 8) == 0x10765432 ? 1 : -1];

#if defined(__LP64__) && !defined(_MSC_VER)
char lrotl_0[_lrotl(0xFEDCBA9876543210ULL, 55) == 0x087F6E5D4C3B2A19ULL ? 1 : -1];
char lrotr_0[_lrotr(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL ? 1 : -1];
#else
char lrotl_0[_lrotl(0x76543210, 22) == 0x841D950C ? 1 : -1];
char lrotr_0[_lrotr(0x76543210, 22) == 0x50C841D9 ? 1 : -1];
#endif

#endif
