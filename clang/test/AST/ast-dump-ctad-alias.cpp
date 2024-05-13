// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2a -ast-dump %s | FileCheck -strict-whitespace %s

template <typename, typename>
constexpr bool Concept = true;
template<typename T> // depth 0
struct Out {
  template<typename U> // depth 1
  struct Inner {
    U t;
  };

  template<typename V> // depth1
  requires Concept<T, V>
  Inner(V) -> Inner<V>;
};

template <typename X>
struct Out2 {
  template<typename Y> // depth1
  using AInner = Out<int>::Inner<Y>;
};
Out2<double>::AInner t(1.0);

// Verify that the require-clause of alias deduction guide is transformed correctly:
//   - Occurrence T should be replaced with `int`;
//   - Occurrence V should be replaced with the Y with depth 1
//
// CHECK:      |   `-FunctionTemplateDecl {{.*}} <deduction guide for AInner>
// CHECK-NEXT: |     |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 Y
// CHECK-NEXT: |     |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'Concept' 
// CHECK-NEXT: |     | |-TemplateArgument type 'int'
// CHECK-NEXT: |     | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     | `-TemplateArgument type 'type-parameter-1-0'
// CHECK-NEXT: |     |   `-TemplateTypeParmType {{.*}} 'type-parameter-1-0' dependent depth 1 index 0
// CHECK-NEXT: |     |-CXXDeductionGuideDecl {{.*}} <deduction guide for AInner> 'auto (type-parameter-0-0) -> Inner<type-parameter-0-0>'
// CHECK-NEXT: |     | `-ParmVarDecl {{.*}} 'type-parameter-0-0'
// CHECK-NEXT: |     `-CXXDeductionGuideDecl {{.*}} used <deduction guide for AInner> 'auto (double) -> Inner<double>' implicit_instantiation
// CHECK-NEXT: |       |-TemplateArgument type 'double'
// CHECK-NEXT: |       | `-BuiltinType {{.*}} 'double'
// CHECK-NEXT: |       `-ParmVarDecl {{.*}} 'double'
