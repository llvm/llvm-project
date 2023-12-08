// Test the -mgeneral-regs-only with -mcrc32 option on x86

// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefix=CMD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefix=CMD %s
// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -mavx2 -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -mavx2 -mgeneral-regs-only %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -mavx2 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -mavx2 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-BEFORE %s
// RUN: %clang -target i386-unknown-linux-gnu -mavx2 -mgeneral-regs-only -mcrc32 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mavx2 -mgeneral-regs-only -mcrc32 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s
// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mavx2 -mcrc32 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mavx2 -mcrc32 %s -### 2>&1 | FileCheck --check-prefixes=CMD,CMD-AFTER %s

// RUN: not %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target i386-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mno-crc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target i386-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: %clang -target i386-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -msse4.2 -mgeneral-regs-only -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: not %clang -target i386-unknown-linux-gnu -mavx2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mavx2 -mgeneral-regs-only -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: %clang -target i386-unknown-linux-gnu -mavx2 -mgeneral-regs-only -mcrc32 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mavx2 -mgeneral-regs-only -S -mcrc32 -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-GPR %s
// RUN: %clang -target i386-unknown-linux-gnu -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s
// RUN: %clang -target i386-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mcrc32 -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=IR-AVX2 %s
// RUN: not %clang -target i386-unknown-linux-gnu -mno-crc32 -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang -target x86_64-unknown-linux-gnu -mno-crc32 -mgeneral-regs-only -mavx2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=ERROR %s

// CMD-BEFORE: "-target-feature" "+crc32"
// CMD: "-target-feature" "-x87"
// CMD: "-target-feature" "-mmx"
// CMD: "-target-feature" "-sse"
// CMD-AFTER: "-target-feature" "+crc32"

unsigned int test__crc32b(unsigned int CRC, unsigned char V) {
// CHECK-LABEL: test__crc32b
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
  return __builtin_ia32_crc32qi(CRC, V);
}

// ERROR: error: '__builtin_ia32_crc32qi' needs target feature crc32

// IR-GPR: attributes {{.*}} = { {{.*}} "target-features"="{{.*}}+crc32{{.*}}-avx{{.*}}-avx2{{.*}}-avx512f{{.*}}-sse{{.*}}-sse2{{.*}}-ssse3{{.*}}-x87{{.*}}"
// IR-AVX2: attributes {{.*}} = { {{.*}} "target-features"="{{.*}}+avx{{.*}}+avx2{{.*}}+crc32{{.*}}+sse{{.*}}+sse2{{.*}}+ssse3{{.*}}-avx512f{{.*}}-x87{{.*}}"
