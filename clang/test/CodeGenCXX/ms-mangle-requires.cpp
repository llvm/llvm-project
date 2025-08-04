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
