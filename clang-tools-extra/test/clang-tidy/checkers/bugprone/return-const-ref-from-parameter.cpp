// RUN: %check_clang_tidy %s bugprone-return-const-ref-from-parameter %t -- -- -fno-delayed-template-parsing

using T = int;
using TConst = int const;
using TConstRef = int const&;

template <typename T>
struct Wrapper { Wrapper(T); };

template <typename T>
struct Identity { using type = T; };

template <typename T>
struct ConstRef { using type = const T&; };

namespace invalid {

int const &f1(int const &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: returning a constant reference parameter

int const &f2(T const &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: returning a constant reference parameter

int const &f3(TConstRef a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: returning a constant reference parameter

int const &f4(TConst &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: returning a constant reference parameter

int const &f5(TConst &a) { return true ? a : a; }
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: returning a constant reference parameter
// CHECK-MESSAGES: :[[@LINE-2]]:46: warning: returning a constant reference parameter

template <typename T>
const T& tf1(const T &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: returning a constant reference parameter

template <typename T>
const T& itf1(const T &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: returning a constant reference parameter

template <typename T>
typename ConstRef<T>::type itf2(const T &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:54: warning: returning a constant reference parameter

template <typename T>
typename ConstRef<T>::type itf3(typename ConstRef<T>::type a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:72: warning: returning a constant reference parameter

template <typename T>
const T& itf4(typename ConstRef<T>::type a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:54: warning: returning a constant reference parameter

template <typename T>
const T& itf5(const T &a) { return true ? a : a; }
// CHECK-MESSAGES: :[[@LINE-1]]:43: warning: returning a constant reference parameter
// CHECK-MESSAGES: :[[@LINE-2]]:47: warning: returning a constant reference parameter

void instantiate(const int &param, const float &paramf, int &mut_param, float &mut_paramf) {
        itf1(0);
        itf1(param);
        itf1(paramf);
        itf2(0);
        itf2(param);
        itf2(paramf);
        itf3<int>(0);
        itf3<int>(param);
        itf3<float>(paramf);
        itf4<int>(0);
        itf4<int>(param);
        itf4<float>(paramf);
}

struct C {
    const C& foo(const C&c) { return c; }
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: returning a constant reference parameter
};

const auto Lf1 = [](const T& t) -> const T& { return t; };
// CHECK-MESSAGES: :[[@LINE-1]]:54: warning: returning a constant reference parameter

} // namespace invalid

namespace false_negative_because_dependent_and_not_instantiated {
template <typename T>
typename ConstRef<T>::type tf2(const T &a) { return a; }

template <typename T>
typename ConstRef<T>::type tf3(typename ConstRef<T>::type a) { return a; }

template <typename T>
const T& tf4(typename ConstRef<T>::type a) { return a; }
} // false_negative_because_dependent_and_not_instantiated

namespace valid {

int const &f1(int &a) { return a; }

int const &f2(int &&a) { return a; }

int f1(int const &a) { return a; }

template <typename T>
T tf1(T a) { return a; }

template <typename T>
T tf2(const T a) { return a; }

template <typename T>
T tf3(const T &a) { return a; }

template <typename T>
Identity<T>::type tf4(const T &a) { return a; }

template <typename T>
T itf1(T a) { return a; }

template <typename T>
T itf2(const T a) { return a; }

template <typename T>
T itf3(const T &a) { return a; }

template <typename T>
Wrapper<T> itf4(const T& a) { return a; }

template <typename T>
const T& itf5(T& a) { return a; }

template <typename T>
T itf6(T& a) { return a; }

void instantiate(const int &param, const float &paramf, int &mut_param, float &mut_paramf) {
        itf1(0);
        itf1(param);
        itf1(paramf);
        itf2(0);
        itf2(param);
        itf2(paramf);
        itf3(0);
        itf3(param);
        itf3(paramf);
        itf2(0);
        itf2(param);
        itf2(paramf);
        itf3(0);
        itf3(param);
        itf3(paramf);
        itf4(param);
        itf4(paramf);
        itf5(mut_param);
        itf5(mut_paramf);
        itf6(mut_param);
        itf6(mut_paramf);
}

template<class T>
void f(const T& t) {
    const auto get = [&t] -> const T& { return t; };
    return T{};
}

const auto Lf1 = [](T& t) -> const T& { return t; };

} // namespace valid

namespace overload {

int const &overload_base(int const &a) { return a; }
int const &overload_base(int &&a);

int const &overload_ret_type(int const &a) { return a; }
void overload_ret_type(int &&a);

int const &overload_params1(int p1, int const &a) { return a; }
int const & overload_params1(int p1, int &&a);

int const &overload_params2(int p1, int const &a, int p2) { return a; }
int const &overload_params2(int p1, int &&a, int p2);

int const &overload_params3(T p1, int const &a, int p2) { return a; }
int const &overload_params3(int p1, int &&a, T p2);

int const &overload_params_const(int p1, int const &a, int const p2) { return a; }
int const &overload_params_const(int const p1, int &&a, int p2);

int const &overload_params_difference1(int p1, int const &a, int p2) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:79: warning: returning a constant reference parameter
int const &overload_params_difference1(long p1, int &&a, int p2);

int const &overload_params_difference2(int p1, int const &a, int p2) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:79: warning: returning a constant reference parameter
int const &overload_params_difference2(int p1, int &&a, long p2);

int const &overload_params_difference3(int p1, int const &a, int p2) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:79: warning: returning a constant reference parameter
int const &overload_params_difference3(int p1, long &&a, int p2);

} // namespace overload

namespace gh117696 {
namespace use_lifetime_bound_attr {
int const &f(int const &a [[clang::lifetimebound]]) { return a; }
} // namespace use_lifetime_bound_attr
} // namespace gh117696


namespace lambda {
using T = const int &;
using K = const float &;
T inner_valid_lambda(T a) {
  [&]() -> T { return a; };
  return a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: returning a constant reference parameter
}
T inner_invalid_lambda(T a) {
  [&](T a) -> T { return a; };
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: returning a constant reference parameter
  return a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: returning a constant reference parameter
}
T inner_invalid_lambda2(T a) {
  [&](K a) -> K { return a; };
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: returning a constant reference parameter
  return a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: returning a constant reference parameter
}
} // namespace lambda
