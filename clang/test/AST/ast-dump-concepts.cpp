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

template <typename... Ts>
concept variadic_concept = true;

template <typename T>
struct Foo {
  // CHECK:      TemplateTypeParmDecl {{.*}} referenced Concept {{.*}} 'binary_concept'
  // CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} <col:13, col:31> 'bool' Concept {{.*}} 'binary_concept'
  // CHECK-NEXT:   |-ImplicitConceptSpecializationDecl {{.*}} <line:13:9> col:9
  // CHECK-NEXT:   | |-TemplateArgument type 'type-parameter-1-0'  
  // CHECK-NEXT:   | | `-TemplateTypeParmType {{.*}} 'type-parameter-1-0' dependent {{.*}}depth 1 index 0
  // CHECK-NEXT:   | `-TemplateArgument type 'int'
  // CHECK-NEXT:   |   `-BuiltinType {{.*}} 'int'
  // CHECK-NEXT:   |-TemplateArgument {{.*}} type 'R'
  // CHECK-NEXT:   | `-TemplateTypeParmType {{.*}} 'R'
  // CHECK-NEXT:   |   `-TemplateTypeParm {{.*}} 'R'
  // CHECK-NEXT:   `-TemplateArgument {{.*}} type 'int'
  // CHECK-NEXT:     `-BuiltinType {{.*}} 'int'
  template <binary_concept<int> R>
  Foo(R);

  // CHECK:      TemplateTypeParmDecl {{.*}} referenced Concept {{.*}} 'unary_concept'
  // CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} <col:13> 'bool'
  // CHECK-NEXT:   |-ImplicitConceptSpecializationDecl {{.*}} <line:10:9> col:9
  // CHECK-NEXT:   | `-TemplateArgument type 'type-parameter-1-0'
  // CHECK-NEXT:   |   `-TemplateTypeParmType {{.*}} 'type-parameter-1-0' dependent {{.*}}depth 1 index 0
  template <unary_concept R>
  Foo(R);

  // CHECK:      FunctionTemplateDecl {{.*}} <line:[[@LINE+1]]:3, line:[[@LINE+2]]:39> {{.*}} Foo<T>
  template <typename R>
  Foo(R, int) requires unary_concept<R>;

  // CHECK:      FunctionTemplateDecl {{.*}} <line:[[@LINE+1]]:3, line:[[@LINE+3]]:3> {{.*}} Foo<T>
  template <typename R>
  Foo(R, char) requires unary_concept<R> {
  }

  // CHECK: CXXFoldExpr {{.*}} <col:13, col:29>
  template <variadic_concept... Ts>
  Foo();

  // CHECK: CXXFoldExpr {{.*}} <col:13, col:34>
  template <variadic_concept<int>... Ts>
  Foo();
};
