// RUN: %clang_cc1 -std=c++20 -triple=x86_64-windows-msvc -Wno-defaulted-function-deleted -fms-compatibility -fms-extensions -emit-llvm %s -o - | FileCheck %s

namespace CWG2369_Regression {

template <class, class Up>
using compare_three_way_result_t = Up::type;

struct sfinae_assign_base {};

template <class Tp>
concept is_derived_from_optional =
    requires(Tp param) { []<class Up>(Up) {}(param); };

template <class Tp, class Up>
  requires(is_derived_from_optional<Up>)
compare_three_way_result_t<Tp, Up> operator<=>(Tp, Up);

struct RuntimeModeArgs {
  auto operator<=>(const RuntimeModeArgs &) const = default;
  sfinae_assign_base needs_admin;
};

RuntimeModeArgs foo() {
  return {};
}

// CHECK: ?foo@CWG2369_Regression@@YA?AURuntimeModeArgs@1@XZ

}
