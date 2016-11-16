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

// CHECK: namespace x {
// CHECK-NEXT: namespace y {
namespace na {
namespace nb {
void f(function<void(int)> func, int param) { func(param); }
void g() { f([](int x) {}, 1); }
// CHECK: } // namespace y
// CHECK-NEXT: } // namespace x
}
}
