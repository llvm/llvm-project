// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ast-dump -ast-dump-filter=dependent_order %s | FileCheck %s --check-prefix=AST

namespace std {
enum class __order : signed char { less = -1, equal = 0, greater = 1 };

struct strong_ordering {
  __order value;

  constexpr explicit strong_ordering(__order value) : value(value) {}

  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less(__order::less);
inline constexpr strong_ordering strong_ordering::equal(__order::equal);
inline constexpr strong_ordering strong_ordering::greater(__order::greater);
} // namespace std

template <class T, class U>
constexpr std::strong_ordering dependent_order() {
  return __builtin_type_order(T, U);
}

// PRINT: template <class T, class U> constexpr std::strong_ordering dependent_order()
// PRINT: return __builtin_type_order(T, U);

// AST-LABEL: FunctionTemplateDecl {{.*}} dependent_order
// AST: BuiltinTypeOrderExpr {{.*}} 'std::strong_ordering'
// AST-NEXT: |-TemplateTypeParmType {{.*}} 'T' dependent
// AST-NEXT: | `-TemplateTypeParm {{.*}} 'T'
// AST-NEXT: `-TemplateTypeParmType {{.*}} 'U' dependent
// AST-NEXT:   `-TemplateTypeParm {{.*}} 'U'
