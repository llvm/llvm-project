// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <x86intrin.h>
#include "builtin_test_helpers.h"

unsigned int test_castf32_u32 (float __A){
  // CHECK-LABEL: test_castf32_u32
  // CHECK: %{{.*}} = load i32, ptr %{{.*}}, align 4
  return _castf32_u32(__A);
}
TEST_CONSTEXPR(_castf32_u32(-0.0f) == 0x80000000);

unsigned long long test_castf64_u64 (double __A){
  // CHECK-LABEL: test_castf64_u64
  // CHECK: %{{.*}} = load i64, ptr %{{.*}}, align 8
  return _castf64_u64(__A);
}
TEST_CONSTEXPR(_castf64_u64(-0.0) == 0x8000000000000000);

float test_castu32_f32 (unsigned int __A){
  // CHECK-LABEL: test_castu32_f32
  // CHECK: %{{.*}} = load float, ptr %{{.*}}, align 4
  return _castu32_f32(__A);
}
TEST_CONSTEXPR(_castu32_f32(0x3F800000) == +1.0f);

double test_castu64_f64 (unsigned long long __A){
  // CHECK-LABEL: test_castu64_f64
  // CHECK: %{{.*}} = load double, ptr %{{.*}}, align 8
  return _castu64_f64(__A);
}
TEST_CONSTEXPR(_castu64_f64(0xBFF0000000000000ULL) == -1.0);
