// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-type-traits %t -check-suffixes=',MACRO'
// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-type-traits %t -- \
// RUN:   -config='{CheckOptions: {modernize-type-traits.IgnoreMacros: true}}'

namespace std {
template <class> struct remove_cv {
  using type = int;
};
template <class T>
using remove_cv_t = typename remove_cv<T>::type; // NOLINT

template <class> struct remove_reference {
  using type = int;
};
template <class T>
using remove_reference_t = typename remove_reference<T>::type; // NOLINT
}

using foo = std::remove_cv_t<std::remove_reference_t<int>>;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use c++20 type alias
// CHECK-FIXES: using foo = std::remove_cvref_t<int>;

std::remove_cv_t<std::remove_reference_t<int>> var;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use c++20 type alias
// CHECK-FIXES: std::remove_cvref_t<int> var;

template<class=std::remove_cv_t<std::remove_reference_t<int>>> struct Foo {};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use c++20 type alias
// CHECK-FIXES: template<class=std::remove_cvref_t<int>> struct Foo {};

#define RM_CV(T) std::remove_cv_t<T>
#define RM_REF(T) std::remove_reference_t<T>
#define RM_CVREF(T) std::remove_cv_t<std::remove_reference_t<T>>
template<class=RM_CV(std::remove_reference_t<int>>) struct M0 {};
// CHECK-MESSAGES-MACRO: :[[@LINE-1]]:16: warning: use c++20 type alias
template<class=std::remove_cv_t<RM_REF(int)>> struct M1 {};
// CHECK-MESSAGES-MACRO: :[[@LINE-1]]:16: warning: use c++20 type alias
template<class=RM_CVREF(int)> struct M2 {};
// CHECK-MESSAGES-MACRO: :[[@LINE-1]]:16: warning: use c++20 type alias
