// RUN: %check_clang_tidy -std=c++11-or-later %s readability-trailing-comma %t

enum class SingleLine2 { X1, Y1 = 1, };
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: enum should not have a trailing comma
// CHECK-FIXES: enum class SingleLine2 { X1, Y1 = 1 };

enum EnumWithAttrs {
  E1 [[deprecated]] = 1,
  E2 [[deprecated]]
};
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: enum should have a trailing comma
// CHECK-FIXES: enum EnumWithAttrs {
// CHECK-FIXES-NEXT:   E1 {{\[\[}}deprecated{{\]\]}} = 1,
// CHECK-FIXES-NEXT:   E2 {{\[\[}}deprecated{{\]\]}},
// CHECK-FIXES-NEXT: };

enum EnumWithAttrs2 {
  E3 [[deprecated]] = 1,
  E4 [[deprecated]] = 2
};
// CHECK-MESSAGES: :[[@LINE-2]]:24: warning: enum should have a trailing comma
// CHECK-FIXES: enum EnumWithAttrs2 {
// CHECK-FIXES-NEXT:   E3 {{\[\[}}deprecated{{\]\]}} = 1,
// CHECK-FIXES-NEXT:   E4 {{\[\[}}deprecated{{\]\]}} = 2,
// CHECK-FIXES-NEXT: };

enum EnumWithAttrsTrailing {
  E5 [[deprecated]] = 1,
  E6 [[deprecated]],
};

enum SingleLineAttrs { E7 [[deprecated]], E8 [[deprecated]] [[deprecated  ]] , };
// CHECK-MESSAGES: :[[@LINE-1]]:78: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleLineAttrs { E7 {{\[\[}}deprecated{{\]\]}}, E8 {{\[\[}}deprecated{{\]\]}} {{\[\[}}deprecated  {{\]\]}}  };

enum SingleLineAttrs2 { E9 [[deprecated]], E10 [[deprecated]] = 1, };
// CHECK-MESSAGES: :[[@LINE-1]]:66: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleLineAttrs2 { E9 {{\[\[}}deprecated{{\]\]}}, E10 {{\[\[}}deprecated{{\]\]}} = 1 };

// Template pack expansions - no warnings
template <typename T, typename... Ts>
struct Pack {
  int values[sizeof...(Ts) + 1] = {sizeof(T), sizeof(Ts)...};
};

Pack<int> one;
Pack<int, double> two;
Pack<int, double, char> three;

template <typename... Ts>
struct PackSingle {
  int values[sizeof...(Ts)] = {sizeof(Ts)...};
};

PackSingle<int> p1;
PackSingle<int, double, char> p3;
