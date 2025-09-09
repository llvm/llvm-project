// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2a -ast-dump -ast-dump-decl-types -ast-dump-filter Foo %s | FileCheck -strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-unknown -include-pch %t \
// RUN: -ast-dump-all -ast-dump-decl-types -ast-dump-filter Foo /dev/null \
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
  // CHECK-NEXT:   | |-TemplateArgument type 'R'
  // CHECK-NEXT:   | | `-TemplateTypeParmType {{.*}} 'R' dependent {{.*}}depth 1 index 0
  // CHECK-NEXT:   | |   `-TemplateTypeParm {{.*}} 'R'
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
  // CHECK-NEXT:   | `-TemplateArgument type 'R'
  // CHECK-NEXT:   |   `-TemplateTypeParmType {{.*}} 'R' dependent {{.*}}depth 1 index 0
  // CHECK-NEXT:   |     `-TemplateTypeParm {{.*}} 'R'
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
  
  // CHECK:InjectedClassNameType
  // CHECK-NEXT: CXXRecord {{.*}} 'Foo'
};

namespace GH82628 {
namespace ns {

template <typename T>
concept C = true;

} // namespace ns

using ns::C;

// CHECK:     ConceptDecl {{.*}} Foo
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} UsingShadow {{.*}} 'C'
template <typename T>
concept Foo = C<T>;

// CHECK: TemplateTypeParmDecl {{.*}} Concept {{.*}} 'C' (UsingShadow {{.*}} 'C')
// CHECK: QualType
// CHECK-NEXT: `-BuiltinType {{.*}} 'bool'
template <C T>
constexpr bool FooVar = false;

// CHECK: ConceptSpecializationExpr {{.*}} UsingShadow {{.*}} 'C'
// CHECK: QualType
// CHECK-NEXT: `-BuiltinType {{.*}} 'bool'
template <typename T> requires C<T>
constexpr bool FooVar2 = true;

// CHECK: SimpleRequirement
// CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} UsingShadow {{.*}} 'C'
// CHECK: QualType
// CHECK-NEXT: `-BuiltinType {{.*}} 'bool'
template <typename T> requires requires (T) { C<T>; }
constexpr bool FooVar3 = true;

// CHECK: NonTypeTemplateParmDecl
// CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} UsingShadow {{.*}} 'C'
// CHECK: QualType
// CHECK-NEXT: `-BuiltinType {{.*}} 'bool'
template <C auto T>
constexpr bool FooVar4 = bool(T());

// CHECK: FunctionTemplateDecl
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} Concept {{.*}} 'C' (UsingShadow {{.*}} 'C') depth 0 index 0 ... T
// CHECK: NonTypeTemplateParmDecl {{.*}} depth 0 index 1 U
// CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} UsingShadow {{.*}} 'C'
// CHECK: |-TemplateTypeParmDecl {{.*}} Concept {{.*}} 'C' (UsingShadow {{.*}} 'C') depth 0 index 2 V:auto
// CHECK: FunctionProtoType
// CHECK: `-Concept {{.*}} 'C'
// CHECK: `-TemplateTypeParm {{.*}} 'V:auto'
template <C... T, C auto U>
auto FooFunc(C auto V) -> C decltype(auto) {
  // FIXME: TypeLocs inside of the function body cannot be dumped via -ast-dump for now.
  // See clang-tools-extra/clangd/unittests/SelectionTests.cpp:SelectionTest.UsingConcepts for their checkings.
  C auto W = V;
  return W;
}

}

namespace constraint_auto_params {

template <class T, class K>
concept C = true;

template<class T>
void g(C<T> auto Foo) {}

// CHECK: TemplateTypeParmDecl {{.*}} depth 0 index 1 Foo:auto
// CHECK-NEXT: `-ConceptSpecializationExpr {{.*}} <col:8, col:11>

}
