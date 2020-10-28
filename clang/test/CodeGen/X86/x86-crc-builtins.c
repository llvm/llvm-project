// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -ffreestanding %s -triple=i686-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

unsigned int test__crc32b(unsigned int CRC, unsigned char V) {
// CHECK-LABEL: test__crc32b
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
  return __crc32b(CRC, V);
}

unsigned int test__crc32w(unsigned int CRC, unsigned short V) {
// CHECK-LABEL: test__crc32w
// CHECK: call i32 @llvm.x86.sse42.crc32.32.16(i32 %{{.*}}, i16 %{{.*}})
  return __crc32w(CRC, V);
}

unsigned int test__crc32d(unsigned int CRC, unsigned int V) {
// CHECK-LABEL: test__crc32d
// CHECK: call i32 @llvm.x86.sse42.crc32.32.32(i32 %{{.*}}, i32 %{{.*}})
  return __crc32d(CRC, V);
}

#ifdef __x86_64__
unsigned long long test__crc32q(unsigned long long CRC, unsigned long long V) {
// CHECK64-LABEL: test__crc32q
// CHECK64: call i64 @llvm.x86.sse42.crc32.64.64(i64 %{{.*}}, i64 %{{.*}})
  return __crc32q(CRC, V);
}
#endif
