// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck %s

using size_t = unsigned long long;

// CHECK: LoopHintAttr {{.*}} Implicit unroll UnrollCount Numeric
// CHECK: LoopHintAttr {{.*}} Implicit unroll UnrollCount Numeric
// CHECK: LoopHintAttr {{.*}} Implicit unroll Unroll Disable
// CHECK: LoopHintAttr {{.*}} Implicit unroll Unroll Disable
template <bool Flag>
int value_dependent(int n) {
  constexpr int N = 100;
  auto init = [=]() { return Flag ? n : 0UL; };
  auto cond = [=](size_t ix) { return Flag ? ix != 0 : ix < 10; };
  auto iter = [=](size_t ix) {
    return Flag ? ix & ~(1ULL << __builtin_clzll(ix)) : ix + 1;
  };

#pragma unroll Flag ? 1 : N
  for (size_t ix = init(); cond(ix); ix = iter(ix)) {
    n *= n;
  }
#pragma unroll Flag ? 0 : N
  for (size_t ix = init(); cond(ix); ix = iter(ix)) {
    n *= n;
  }
  return n;
}

void test_value_dependent(int n) {
  value_dependent<true>(n);
}
