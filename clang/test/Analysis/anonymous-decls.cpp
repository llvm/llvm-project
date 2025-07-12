// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++20 %s 2>&1 | FileCheck %s

struct A {
  static A a;
  char b;
  friend bool operator==(A, A) = default;
};
bool _ = A() == A::a;

// FIXME: steps 1 and 5 show anonymous function parameters are
// not handled correctly.

// CHECK-LABEL: bool operator==(A, A) noexcept = default
// CHECK-NEXT: [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK:      [B1]
// CHECK-NEXT:    1: function-parameter-0-0
// CHECK-NEXT:    2: [B1.1].b
// CHECK-NEXT:    3: [B1.2] (ImplicitCastExpr, LValueToRValue, char)
// CHECK-NEXT:    4: [B1.3] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    5: function-parameter-0-1
// CHECK-NEXT:    6: [B1.5].b
// CHECK-NEXT:    7: [B1.6] (ImplicitCastExpr, LValueToRValue, char)
// CHECK-NEXT:    8: [B1.7] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    9: [B1.4] == [B1.8]
// CHECK-NEXT:   10: return [B1.9];
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:      [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1

namespace std {
template <class> struct iterator_traits;
template <class, class> struct pair;
template <class _Tp> struct iterator_traits<_Tp *> {
  typedef _Tp &reference;
};
template <long, class> struct tuple_element;
template <class> struct tuple_size;
template <class _T1, class _T2> struct tuple_size<pair<_T1, _T2>> {
  static const int value = 2;
};
template <class _T1, class _T2> struct tuple_element<0, pair<_T1, _T2>> {
  using type = _T1;
};
template <class _T1, class _T2> struct tuple_element<1, pair<_T1, _T2>> {
  using type = _T2;
};
template <long _Ip, class _T1, class _T2>
tuple_element<_Ip, pair<_T1, _T2>>::type get(pair<_T1, _T2> &);
struct __wrap_iter {
  iterator_traits<pair<int, int> *>::reference operator*();
  void operator++();
};
bool operator!=(__wrap_iter, __wrap_iter);
struct vector {
  __wrap_iter begin();
  __wrap_iter end();
};
} // namespace std
int main() {
  std::vector v;
  for (auto &[a, b] : v)
    ;
}

// FIXME: On steps 8 and 14, a decomposition is referred by name, which they never have.

// CHECK-LABEL: int main()
// CHECK:      [B3]
// CHECK-NEXT:   1: operator*
// CHECK-NEXT:   2: [B3.1] (ImplicitCastExpr, FunctionToPointerDecay, iterator_traits<pair<int, int> *>::reference (*)(void))
// CHECK-NEXT:   3: __begin1
// CHECK-NEXT:   4: * [B3.3] (OperatorCall)
// CHECK-NEXT:   5: auto &;
// CHECK-NEXT:   6: get<0UL>
// CHECK-NEXT:   7: [B3.6] (ImplicitCastExpr, FunctionToPointerDecay, tuple_element<0L, pair<int, int> >::type (*)(pair<int, int> &))
// CHECK-NEXT:   8: decomposition-a-b
// CHECK-NEXT:   9: [B3.7]([B3.8])
// CHECK-NEXT:  10: [B3.9]
// CHECK-NEXT:  11: std::tuple_element<0UL, std::pair<int, int>>::type a = get<0UL>(decomposition-a-b);
// CHECK-NEXT:  12: get<1UL>
// CHECK-NEXT:  13: [B3.12] (ImplicitCastExpr, FunctionToPointerDecay, tuple_element<1L, pair<int, int> >::type (*)(pair<int, int> &))
// CHECK-NEXT:  14: decomposition-a-b
// CHECK-NEXT:  15: [B3.13]([B3.14])
// CHECK-NEXT:  16: [B3.15]
// CHECK-NEXT:  17: std::tuple_element<1UL, std::pair<int, int>>::type b = get<1UL>(decomposition-a-b);
// CHECK-NEXT:   Preds (1): B1
// CHECK-NEXT:   Succs (1): B2
