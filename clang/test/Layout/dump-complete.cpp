// RUN: %clang_cc1 -fsyntax-only -fdump-record-layouts-complete %s | FileCheck %s

struct a {
  int x;
};

struct b {
  char y;
} foo;

class c {};

class d;

template <typename>
struct s {
  int x;
};

template <typename T>
struct ts {
  T x;
};

template <>
struct ts<void> {
  float f;
};

void f() {
  ts<int> a;
  ts<double> b;
  ts<void> c;
}

namespace gh83671 {
template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef integral_constant type;
};

template <bool _Val>
using _BoolConstant = integral_constant<bool, _Val>;

template <class _Tp, class _Up>
struct is_same : _BoolConstant<__is_same(_Tp, _Up)> {};

template < class _Tp >
class numeric_limits {};

template < class _Tp >
class numeric_limits< const _Tp > : public numeric_limits< _Tp > {};
}

namespace gh83684 {
template <class Pointer>
struct AllocationResult {
  Pointer ptr = nullptr;
  int count = 0;
};
}

// CHECK:          0 | struct a
// CHECK:          0 | struct b
// CHECK:          0 | class c
// CHECK:          0 | struct ts<void>
// CHECK-NEXT:     0 |   float
// CHECK:          0 | struct ts<int>
// CHECK:          0 | struct ts<double>
// CHECK-NOT:      0 | class d
// CHECK-NOT:      0 | struct s
// CHECK-NOT:      0 | struct AllocationResult
