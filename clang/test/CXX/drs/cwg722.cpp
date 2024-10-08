// RUN: %clang_cc1 -std=c++98 %s -verify -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -verify -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -verify -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++26 %s -verify -pedantic-errors -ast-dump | FileCheck %s

// expected-no-diagnostics
// cwg722: 20

#if __cplusplus >= 201103L
namespace std {
  using nullptr_t = decltype(nullptr);
}

void f(std::nullptr_t...);
std::nullptr_t g();
void h() {
  std::nullptr_t np;
  const std::nullptr_t cnp = nullptr;
  extern int i;
  f(
    nullptr,
    nullptr, np, cnp,
    static_cast<std::nullptr_t>(np),
    g(),
    __builtin_bit_cast(std::nullptr_t, static_cast<void*>(&i))
  );
// CHECK:      `-CallExpr {{.+}} 'void'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void (*)(std::nullptr_t, ...)' <FunctionToPointerDecay>
// CHECK-NEXT:  | `-DeclRefExpr {{.+}} 'void (std::nullptr_t, ...)' lvalue Function {{.+}} 'f' 'void (std::nullptr_t, ...)'
// CHECK-NEXT:  |-CXXNullPtrLiteralExpr {{.+}} 'std::nullptr_t'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:  | `-CXXNullPtrLiteralExpr {{.+}} 'std::nullptr_t'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:  | `-DeclRefExpr {{.+}} 'std::nullptr_t' lvalue Var {{.+}} 'np' 'std::nullptr_t'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:  | `-DeclRefExpr {{.+}} 'const std::nullptr_t' lvalue Var {{.+}} 'cnp' 'const std::nullptr_t'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:  | `-CXXStaticCastExpr {{.+}} 'std::nullptr_t' static_cast<std::nullptr_t> <NoOp>
// CHECK-NEXT:  |   `-ImplicitCastExpr {{.+}} 'std::nullptr_t' <NullToPointer> part_of_explicit_cast
// CHECK-NEXT:  |     `-DeclRefExpr {{.+}} 'std::nullptr_t' lvalue Var {{.+}} 'np' 'std::nullptr_t'
// CHECK-NEXT:  |-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:  | `-CallExpr {{.+}} 'std::nullptr_t'
// CHECK-NEXT:  |   `-ImplicitCastExpr {{.+}} 'std::nullptr_t (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:  |     `-DeclRefExpr {{.+}} 'std::nullptr_t ()' lvalue Function {{.+}} 'g' 'std::nullptr_t ()'
// CHECK-NEXT:  `-ImplicitCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT:    `-BuiltinBitCastExpr {{.+}} 'std::nullptr_t' <LValueToRValueBitCast>
// CHECK-NEXT:      `-MaterializeTemporaryExpr {{.+}} 'void *' xvalue
// CHECK-NEXT:        `-CXXStaticCastExpr {{.+}} 'void *' static_cast<void *> <NoOp>
// CHECK-NEXT:          `-ImplicitCastExpr {{.+}} 'void *' <BitCast> part_of_explicit_cast
// CHECK-NEXT:            `-UnaryOperator {{.+}} 'int *' prefix '&' cannot overflow
// CHECK-NEXT:              `-DeclRefExpr {{.+}} 'int' lvalue Var {{.+}} 'i' 'int'
}
#endif
