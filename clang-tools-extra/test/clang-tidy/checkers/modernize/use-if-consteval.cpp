// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-if-consteval %t

namespace std {
constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
} // namespace std

namespace mine {
constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
} // namespace mine

namespace alias = std;

#define ICE_CALL() std::is_constant_evaluated()
#define IF_ICE_HEADER if (std::is_constant_evaluated())
#define IF_ONLY if
#define RETURN_ONE() return 1;
#define RETURN_THREE() return 3;

bool runtime_predicate();

int direct() {
  if (std::is_constant_evaluated())
    return 1;
  else
    return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()' [modernize-use-if-consteval]
  // CHECK-FIXES:      if consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT:   return 2;
  // CHECK-FIXES-NEXT: }
}

int direct_global() {
  if (::std::is_constant_evaluated()) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int compact_spacing() {
  if(std::is_constant_evaluated()) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:6: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int using_decl() {
  using std::is_constant_evaluated;
  if (is_constant_evaluated()) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int using_namespace() {
  using namespace std;
  if (is_constant_evaluated()) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int namespace_alias() {
  if (alias::is_constant_evaluated()) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int negated() {
  if (!std::is_constant_evaluated())
    return 1;
  return 2;
  // CHECK-MESSAGES: :[[@LINE-3]]:8: warning: use 'if !consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if !consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 2;
}

int negated_alternative_token() {
  if (not std::is_constant_evaluated())
    return 1;
  return 2;
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use 'if !consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if !consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 2;
}

int extra_parens() {
  if ((((std::is_constant_evaluated())))) {
    return 1;
  }
  return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:10: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

template <typename T>
int templated() {
  if (std::is_constant_evaluated()) {
    return sizeof(T);
  }
  return 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

template int templated<int>();
template int templated<long>();

auto Lambda = [] {
  if (std::is_constant_evaluated())
    return 1;
  return 2;
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 2;
};

int attributed_then() {
  if (std::is_constant_evaluated())
    [[likely]] return 1;
  return 0;
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
  // CHECK-FIXES-NEXT:   {{[[][[]}}likely{{[]][]]}} return 1;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 0;
}

int labeled_then() {
  if (std::is_constant_evaluated())
  labeled_then:
    return 1;
  return 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
  // CHECK-FIXES-NEXT:   labeled_then:
  // CHECK-FIXES-NEXT:     return 1;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 0;
}

int else_if_chain(int Value) {
  if (Value == 0)
    return 0;
  else if (std::is_constant_evaluated())
    return 1;
  else
    return 2;
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      else if consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: } else {
  // CHECK-FIXES-NEXT:   return 2;
  // CHECK-FIXES-NEXT: }
}

int outer_else_if() {
  if (std::is_constant_evaluated())
    return 1;
  else if (runtime_predicate())
    return 2;
  return 0;
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: } else { if (runtime_predicate())
  // CHECK-FIXES-NEXT:   return 2;
  // CHECK-FIXES-NEXT: }
  // CHECK-FIXES-NEXT:   return 0;
}

int macro_header_safe() {
  if (ICE_CALL()) {
    return 1;
  } else {
    return 2;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if consteval {
}

int with_init() {
  if (int X = 0; std::is_constant_evaluated()) {
    return X + 1;
  }
  return 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:18: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if (int X = 0; std::is_constant_evaluated()) {
}

int with_condition_variable() {
  if (bool B = std::is_constant_evaluated())
    return B ? 1 : 2;
  else
    return 3;
  // CHECK-MESSAGES: :[[@LINE-4]]:16: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if (bool B = std::is_constant_evaluated())
  // CHECK-FIXES-NEXT:   return B ? 1 : 2;
  // CHECK-FIXES-NEXT: else
  // CHECK-FIXES-NEXT:   return 3;
}

int macro_header_unsafe() {
  IF_ICE_HEADER {
    return 1;
  }
  return 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      IF_ICE_HEADER {
}

int macro_if_token_unsafe() {
  IF_ONLY (std::is_constant_evaluated()) {
    return 1;
  }
  return 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      IF_ONLY consteval {
}

int macro_body_unsafe() {
  if (std::is_constant_evaluated())
    RETURN_ONE()
  return 2;
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if (std::is_constant_evaluated())
  // CHECK-FIXES-NEXT:   RETURN_ONE()
  // CHECK-FIXES-NEXT:   return 2;
}

int macro_else_unsafe() {
  if (std::is_constant_evaluated())
    return 1;
  else
    RETURN_THREE()
  return 4;
  // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: use 'if consteval' instead of checking 'std::is_constant_evaluated()'
  // CHECK-FIXES:      if (std::is_constant_evaluated())
  // CHECK-FIXES-NEXT:   return 1;
  // CHECK-FIXES-NEXT: else
  // CHECK-FIXES-NEXT:   RETURN_THREE()
  // CHECK-FIXES-NEXT:   return 4;
}

int not_std() {
  if (mine::is_constant_evaluated()) {
    return 1;
  }
  return 0;
}

int composite_conditions() {
  if (std::is_constant_evaluated() && runtime_predicate()) {
    return 1;
  }
  if (!!std::is_constant_evaluated()) {
    return 2;
  }
  return 0;
}

int if_constexpr() {
  if constexpr (std::is_constant_evaluated()) {
    return 1;
  }
  return 2;
}

int already_if_consteval() {
  if consteval {
    return 1;
  } else {
    return 2;
  }
}

int already_if_not_consteval() {
  if !consteval {
    return 1;
  } else {
    return 2;
  }
}

template <typename T>
concept HasICE = requires {
  std::is_constant_evaluated();
};

using ICEPtr = decltype(std::is_constant_evaluated()) *;
ICEPtr Ptr = nullptr;
