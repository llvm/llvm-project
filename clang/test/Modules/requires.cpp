// RUN: mkdir -p %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -emit-pch -o %t/foo.pch

template <class, class> class expected;
template <class _Tp, class _Err>
  requires true
class expected<_Tp, _Err> {
  friend void swap(expected __x)
    requires requires { __x; }
  {}
};

template <int>
  requires requires { 0; }
using iter_rvalue_reference_t = int;
