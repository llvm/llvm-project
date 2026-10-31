// RUN: %check_clang_tidy -std=c++11,c++14 -check-suffix=COMMON -expect-clang-tidy-error %s modernize-use-noexcept %t
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=COMMON,CXX17 -expect-clang-tidy-error %s modernize-use-noexcept %t

struct S {
  template <typename T>
  static void f() throw(typename T::X);
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:19: error: ISO C++17 does not allow dynamic exception specifications [clang-diagnostic-dynamic-exception-spec]
  // CHECK-MESSAGES-COMMON: :[[@LINE-2]]:19: warning: dynamic exception specification 'throw(typename T::X)' is deprecated; consider using 'noexcept(false)' instead [modernize-use-noexcept]

  typedef decltype(f<S>()) X;
  // CHECK-MESSAGES-COMMON: :[[@LINE-1]]:20: error: exception specification is not available until end of class definition [clang-diagnostic-error]
};
