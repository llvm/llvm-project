// RUN: clang-change-namespace -old_namespace "na::nb" -new_namespace "x::y" --file_pattern ".*" %s -- -std=c++11 | sed 's,// CHECK.*,,' | FileCheck %s

template <class T>
class function;
template <class R, class... ArgTypes>
class function<R(ArgTypes...)> {
public:
  template <typename Functor>
  function(Functor f) {}
  R operator()(ArgTypes...) const {}
};

namespace x {
// CHECK: namespace x {
class X {};
}

namespace na {
namespace nb {
// CHECK: namespace x {
// CHECK-NEXT: namespace y {
void f(function<void(int)> func, int param) { func(param); }
void g() { f([](int x) {}, 1); }

// x::X in function type parameter list will have translation unit context, so
// we simply replace it with fully-qualified name.
using TX = function<x::X(x::X)>;
// CHECK: using TX = function<X(x::X)>;

class A {};
using TA = function<A(A)>;
// CHECK: using TA = function<A(A)>;

// CHECK: } // namespace y
// CHECK-NEXT: } // namespace x
}
}
