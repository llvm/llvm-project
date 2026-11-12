// RUN: %check_clang_tidy -std=c++17-or-later %s readability-misleading-indentation %t -- -- -fno-delayed-template-parsing

namespace PR61435 {

template<int N>
constexpr auto lam_correct = []{
  if constexpr (N == 1) {
  } else {
  }
};

template<int N>
constexpr auto lam_incorrect = []{
  if constexpr (N == 1) {
  }
   else {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:4: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]
};

void test() {
  lam_correct<1>();
  lam_correct<2>();

  lam_incorrect<1>();
  lam_incorrect<2>();
}

}
