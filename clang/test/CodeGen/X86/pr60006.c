// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

// Ensure _mm_test_all_ones macro doesn't reuse argument
__m128i expensive_call();
int pr60006() {
  // CHECK-LABEL: pr60006
  // CHECK: call {{.*}} @expensive_call
  // CHECK-NOT: call {{.*}} @expensive_call
  // CHECK: call i32 @llvm.x86.sse41.ptestc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_all_ones(expensive_call());
}
