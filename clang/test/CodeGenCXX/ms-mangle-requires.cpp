// RUN: %clang_cc1 -std=c++20 -triple=x86_64-windows-msvc -Wno-defaulted-function-deleted -fms-compatibility -fms-extensions -emit-llvm %s -o - | FileCheck %s

namespace CWG2369 {

namespace Regression1 {

template <class, class Up>
using compare_three_way_result_t = Up::type;

struct sfinae_assign_base {};

template <class Tp>
concept is_derived_from_optional =
    requires(Tp param) { []<class Up>(Up) {}(param); };

template <class Tp, class Up>
  requires(is_derived_from_optional<Up> && []<class W>(W) { return true; }(Up()))
compare_three_way_result_t<Tp, Up> operator<=>(Tp, Up);

struct RuntimeModeArgs {
  auto operator<=>(const RuntimeModeArgs &) const = default;
  sfinae_assign_base needs_admin;
};

RuntimeModeArgs foo() {
  return {};
}

// CHECK: ?foo@Regression1@CWG2369@@YA?AURuntimeModeArgs@12@XZ

} // namespace Regression1

namespace Regression2 {

template <class _Tp>
constexpr _Tp * __to_address(_Tp *) {
  return nullptr;
}

template <class _Ip>
concept contiguous_iterator = requires(_Ip __i) { __to_address(__i); };

struct basic_string_view {
  template <contiguous_iterator _It>
  basic_string_view(_It, _It);
};

const char *str;
void sv() { basic_string_view(str, str); }

void m_fn2() {
  const char __trans_tmp_1 = *__to_address(&__trans_tmp_1);
}

// CHECK: define {{.*}} @"??$__to_address@$$CBD@Regression2@CWG2369@@YAPEBDPEBD@Z"

} // namespace Regression2

}

namespace GH147650 {

template <int> int b;
template <int b>
void f()
    requires requires { [] { (void)b; }; } {}
void test() {
    f<42>();
}
// CHECK-LABEL:define {{.*}} void @"??$f@$0CK@@GH147650@@YAXXZ"()
}

namespace GH173160 {

template< typename TypeList, typename Func >
auto for_each_type(Func func) {
  return func;
}

template< class T >
struct X {};

template< class T >
concept Component = requires(T component) {
  for_each_type<typename T::required_type>( []{} );
};

struct EntityRegistry {
  template< Component C >
  auto get_component(const int id) -> X<C> {
    return {};
  }
};

struct A {
  struct required_type {};
};

void foo() {
  EntityRegistry entities;
  entities.get_component<A>(0);
  // CHECK: define {{.*}} @"??$get_component@UA@GH173160@@@EntityRegistry@GH173160@@QEAA?AU?$X@UA@GH173160@@@1@H@Z"
}

}

namespace GH191588 {

template<typename T> concept HasBar = requires(T t) {
  t.bar([](const auto&) {});
};

template<class V, class... F> requires (HasBar<V>)
decltype(auto) testBar(V&& v, F&&... f) { return 0; }

struct WithVariadic {
  template<typename... F> decltype(auto) bar(F&&... f) { return 0; }
};

void caller() {
  testBar(WithVariadic{}, [](auto){});
}

// CHECK: define {{.*}} @"??$testBar@UWithVariadic@GH191588@@V<lambda_0>@?0??caller@2@YAXXZ@@GH191588@@YA?A?<decltype-auto>@@$$QEAUWithVariadic@0@$$QEAV<lambda_0>@?0??caller@0@YAXXZ@@Z"

}
