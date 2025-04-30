// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

namespace LambdaContainingLocalClasses {

template <typename F>
void GH59734() {
  [&](auto param) {
    struct Guard {
      Guard() {
        // Check that we're able to create DeclRefExpr to param at this point.
        static_assert(__is_same(decltype(param), int), "");
      }
      ~Guard() {
        static_assert(__is_same(decltype(param), int), "");
      }
      operator decltype(param)() {
        return decltype(param)();
      }
    };
    Guard guard;
    param = guard;
  }(42);
}

// Guard::Guard():
// CHECK-DAG: define {{.*}} @_ZZZN28LambdaContainingLocalClasses7GH59734IiEEvvENKUlT_E_clIiEEDaS1_EN5GuardC2Ev
// Guard::operator int():
// CHECK-DAG: define {{.*}} @_ZZZN28LambdaContainingLocalClasses7GH59734IiEEvvENKUlT_E_clIiEEDaS1_EN5GuardcviEv
// Guard::~Guard():
// CHECK-DAG: define {{.*}} @_ZZZN28LambdaContainingLocalClasses7GH59734IiEEvvENKUlT_E_clIiEEDaS1_EN5GuardD2Ev

struct S {};

template <class T = void>
auto GH132208 = [](auto param) {
  struct OnScopeExit {
    OnScopeExit() {
      static_assert(__is_same(decltype(param), S), "");
    }
    ~OnScopeExit() {
      static_assert(__is_same(decltype(param), S), "");
    }
    operator decltype(param)() {
      return decltype(param)();
    }
  } pending;

  param = pending;
};

void bar() {
  GH59734<int>();

  GH132208<void>(S{});
}

// OnScopeExit::OnScopeExit():
// CHECK-DAG: define {{.*}} @_ZZNK28LambdaContainingLocalClasses8GH132208IvEMUlT_E_clINS_1SEEEDaS2_EN11OnScopeExitC2Ev
// OnScopeExit::operator S():
// CHECK-DAG: define {{.*}} @_ZZNK28LambdaContainingLocalClasses8GH132208IvEMUlT_E_clINS_1SEEEDaS2_EN11OnScopeExitcvS5_Ev
// OnScopeExit::~OnScopeExit():
// CHECK-DAG: define {{.*}} @_ZZNK28LambdaContainingLocalClasses8GH132208IvEMUlT_E_clINS_1SEEEDaS2_EN11OnScopeExitD2Ev

} // namespace LambdaContainingLocalClasses
