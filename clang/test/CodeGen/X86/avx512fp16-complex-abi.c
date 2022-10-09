// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -target-feature +avx512fp16 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK

// Return value should be passed in <2 x half> so the backend will use xmm0
_Complex _Float16 f16(_Complex _Float16 A, _Complex _Float16 B) {
  // CHECK-LABEL: define{{.*}}<2 x half> @f16(ptr noundef byval({ half, half }) align 4 %{{.*}}, ptr noundef byval({ half, half }) align 4 %{{.*}})
  return A + B;
}
