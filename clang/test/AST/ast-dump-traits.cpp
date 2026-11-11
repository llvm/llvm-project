// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck -strict-whitespace %s

void test_type_trait() {
  // An unary type trait.
  enum E {};
  (void) __is_enum(E);
  // A binary type trait.
  (void) __is_same(int ,float);
  // An n-ary type trait.
  (void) __is_constructible(int, int, int, int);
}

void test_array_type_trait() {
  // An array type trait.
  (void) __array_rank(int[10][20]);
}

void test_expression_trait() {
  // An expression trait.
  (void) __is_lvalue_expr(1);
}

void test_unary_expr_or_type_trait() {
  // Some UETTs.
  (void) sizeof(int);
  (void) alignof(int);
  (void) __alignof(int);
}

namespace std {
struct strong_ordering {
  signed char value;

  constexpr explicit strong_ordering(signed char value) : value(value) {}

  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less{-1};
inline constexpr strong_ordering strong_ordering::equal{0};
inline constexpr strong_ordering strong_ordering::greater{1};
} // namespace std

template <class T, class U>
constexpr std::strong_ordering test_strong_ordering_type_trait() {
  return __builtin_type_order(T, U);
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>{{( <undeserialized declarations>)?}}
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-traits.cpp:10:1, line:18:1> line:10:6{{( imported)?}} test_type_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:24, line:18:1>
// CHECK-NEXT: |   |-DeclStmt {{.*}} <line:12:3, col:12>
// CHECK-NEXT: |   | `-EnumDecl {{.*}} <col:3, col:11> col:8{{( imported)?}} referenced E
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} <line:13:3, col:21> 'void' <ToVoid>
// CHECK-NEXT: |   | `-TypeTraitExpr {{.*}} <col:10, col:21> 'bool' __is_enum
// CHECK-NEXT: |   |   `-EnumType {{.*}} 'E'
// CHECK-NEXT: |   |     `-Enum {{.*}} 'E'
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} <line:15:3, col:30> 'void' <ToVoid>
// CHECK-NEXT: |   | `-TypeTraitExpr {{.*}} <col:10, col:30> 'bool' __is_same
// CHECK-NEXT: |   |   |-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} <line:17:3, col:47> 'void' <ToVoid>
// CHECK-NEXT: |     `-TypeTraitExpr {{.*}} <col:10, col:47> 'bool' __is_constructible
// CHECK-NEXT: |       |-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       `-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:20:1, line:23:1> line:20:6{{( imported)?}} test_array_type_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:30, line:23:1>
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} <line:22:3, col:34> 'void' <ToVoid>
// CHECK-NEXT: |     `-ArrayTypeTraitExpr {{.*}} <col:10, col:34> '__size_t':'unsigned long' __array_rank
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:25:1, line:28:1> line:25:6{{( imported)?}} test_expression_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:30, line:28:1>
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} <line:27:3, col:28> 'void' <ToVoid>
// CHECK-NEXT: |     `-ExpressionTraitExpr {{.*}} <col:10, col:28> 'bool' __is_lvalue_expr
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:30:1, line:35:1> line:30:6{{( imported)?}} test_unary_expr_or_type_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:38, line:35:1>
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} <line:32:3, col:20> 'void' <ToVoid>
// CHECK-NEXT: |   | `-UnaryExprOrTypeTraitExpr {{.*}} <col:10, col:20> '__size_t':'unsigned long' sizeof 'int'
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} <line:33:3, col:21> 'void' <ToVoid>
// CHECK-NEXT: |   | `-UnaryExprOrTypeTraitExpr {{.*}} <col:10, col:21> '__size_t':'unsigned long' alignof 'int'
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} <line:34:3, col:23> 'void' <ToVoid>
// CHECK-NEXT: |     `-UnaryExprOrTypeTraitExpr {{.*}} <col:10, col:23> '__size_t':'unsigned long' __alignof 'int'
// CHECK: `-FunctionTemplateDecl {{.*}} <line:53:1, line:56:1> line:54:32{{( imported)?}} test_strong_ordering_type_trait
// CHECK-NEXT:   |-TemplateTypeParmDecl {{.*}} <line:53:11, col:17> col:17{{( imported)?}} referenced class depth 0 index 0 T
// CHECK-NEXT:   |-TemplateTypeParmDecl {{.*}} <col:20, col:26> col:26{{( imported)?}} referenced class depth 0 index 1 U
// CHECK-NEXT:   `-FunctionDecl {{.*}} <line:54:1, line:56:1> line:54:32{{( imported)?}} constexpr test_strong_ordering_type_trait 'std::strong_ordering ()'
// CHECK-NEXT:     `-CompoundStmt {{.*}} <col:66, line:56:1>
// CHECK-NEXT:       `-ReturnStmt {{.*}} <line:55:3, col:35>
// CHECK-NEXT:         `-TypeTraitExpr {{.*}} <col:10, col:35> 'std::strong_ordering' __builtin_type_order
// CHECK-NEXT:           |-TemplateTypeParmType {{.*}} 'T' dependent{{( imported)?}} depth 0 index 0
// CHECK-NEXT:           | `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT:           `-TemplateTypeParmType {{.*}} 'U' dependent{{( imported)?}} depth 0 index 1
// CHECK-NEXT:             `-TemplateTypeParm {{.*}} 'U'
