// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

using u8 = unsigned char;
using u64 = unsigned long long;

struct Padded {
  u8 Head{};
  u64 Middle{};
  u8 Tail{};
  u8 First[2]{};
  u8 Second[2]{};
  u8 Third[2]{};
  constexpr bool operator==(const Padded &) const = default;
};

template <Padded P> int readFirst() {
  volatile int I = 0;
  return P.First[I];
}

template <Padded P> int readSecond() {
  volatile int I = 0;
  return P.Second[I];
}

constexpr Padded P = [] {
  Padded Result{};
  Result.First[0] = 5;
  Result.Second[0] = 6;
  Result.Third[0] = 7;
  return Result;
}();

int use() { return readFirst<P>() + readSecond<P>(); }

// `First` starts immediately after Head, Middle, and Tail, including the
// padding before Middle: 1 + 7 + 8 + 1 == 17.
// CHECK-LABEL: define {{.*}} @_Z9readFirst
// CHECK: getelementptr {{.*}}(i8, ptr @_ZTAX{{.*}}, i64 17)

// `Second` starts after `First`: 17 + 2 == 19.
// CHECK-LABEL: define {{.*}} @_Z10readSecond
// CHECK: getelementptr {{.*}}(i8, ptr @_ZTAX{{.*}}, i64 19)
