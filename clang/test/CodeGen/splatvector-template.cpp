// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -target-cpu corei7-avx -emit-llvm %s -o - | FileCheck %s

template <typename T, auto N>
using simd = T [[clang::ext_vector_type(N)]];

template <typename T, auto N>
constexpr static simd<T, N> splat(T v) {
  return __builtin_splatvector(v, simd<T, N>);
  // CHECK-LABEL: internal
  // CHECK: store [[TYPE:.*]] %v, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load [[TYPE]], ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <[[SIZE:[0-9]*]] x [[TYPE]]> poison, [[TYPE]] %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <[[SIZE]] x [[TYPE]]> %[[INSERT]], <[[SIZE]] x [[TYPE]]> poison, <[[SIZE]] x [[TYPE]]> zeroinitializer
  // CHECK: ret <[[SIZE]] x [[TYPE]]> %[[SPLAT]]
}

const auto vi32x8 = splat<int, 8U>(123);
