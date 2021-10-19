// Test interaction between -mcrc32 and other SIMD ISA options on x86

// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

// RUN: %clang -target i386-unknown-linux-gnu -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

// RUN: %clang -target i386-unknown-linux-gnu -msse4.2 -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -msse4.2 -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

// RUN: not %clang -target i386-unknown-linux-gnu -mno-crc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mno-crc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s

// RUN: not %clang -target i386-unknown-linux-gnu -msse4.2 -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -msse4.2 -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s

// RUN: not %clang -target i386-unknown-linux-gnu -mcrc32 -mno-crc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mcrc32 -mno-crc32 -msse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s

// RUN: not %clang -target i386-unknown-linux-gnu -mcrc32 -msse4.2 -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mcrc32 -msse4.2 -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s

// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -mno-sse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -mno-sse4.2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

// RUN: %clang -target i386-unknown-linux-gnu -mno-sse4.2 -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-sse4.2 -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-CRC32 %s

unsigned int test__crc32b(unsigned int CRC, unsigned char V) {
// CHECK-LABEL: test__crc32b
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
  return __builtin_ia32_crc32qi(CRC, V);
}

// ERROR: error: '__builtin_ia32_crc32qi' needs target feature crc32

// IR-CRC32: attributes {{.*}} = { {{.*}} "target-features"="{{.*}}+crc32{{.*}}"
