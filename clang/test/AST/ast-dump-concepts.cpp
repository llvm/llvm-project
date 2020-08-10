// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2a -ast-dump -ast-dump-filter Foo %s | FileCheck -strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-unknown -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Foo /dev/null \
// RUN: | FileCheck --strict-whitespace %s

template <typename T>
concept unary_concept = true;

template <typename T, typename U>
concept binary_concept = true;

template <typename T>
struct Foo {
  // CHECK:      TemplateTypeParmDecl {{.*}} referenced Concept {{.*}} 'binary_concept'
  // CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} <col:13, col:31> 'bool' Concept {{.*}} 'binary_concept'
  // CHECK-NEXT:   |-TemplateArgument {{.*}} type 'R'
  // CHECK-NEXT:   | `-TemplateTypeParmType {{.*}} 'R'
  // CHECK-NEXT:   |   `-TemplateTypeParm {{.*}} 'R'
  // CHECK-NEXT:   `-TemplateArgument {{.*}} type 'int'
  // CHECK-NEXT:     `-BuiltinType {{.*}} 'int'
  template <binary_concept<int> R>
  Foo(R);

  // CHECK:      TemplateTypeParmDecl {{.*}} referenced Concept {{.*}} 'unary_concept'
  // CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} <col:13> 'bool'
  template <unary_concept R>
  Foo(R);

  // CHECK:      FunctionTemplateDecl {{.*}} <line:[[@LINE+1]]:3, line:[[@LINE+2]]:39> {{.*}} Foo<T>
  template <typename R>
  Foo(R, int) requires unary_concept<R>;

  // CHECK:      FunctionTemplateDecl {{.*}} <line:[[@LINE+1]]:3, line:[[@LINE+3]]:3> {{.*}} Foo<T>
  template <typename R>
  Foo(R, char) requires unary_concept<R> {
  }
};
