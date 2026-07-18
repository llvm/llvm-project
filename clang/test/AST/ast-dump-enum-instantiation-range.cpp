// RUN: %clang_cc1 -std=c++17 -ast-dump %s | FileCheck %s

// Instantiated enum ranges must include their enumerators.

template <typename T> struct A {
  enum E { X = 1, Y = 2 };
  enum class S : int { P, Q };
  enum class U : int; // Remains undefined.
};

A<int> a;
A<int>::S s = A<int>::S::P; // Instantiate S's definition.

template <typename T> int f() {
  enum L { M = sizeof(T) };
  return M;
}
int force_f = f<char>();

// Pattern enums.
// CHECK: EnumDecl {{.*}} <line:[[@LINE-15]]:3, col:25> col:8 E
// CHECK: EnumDecl {{.*}} <line:[[@LINE-15]]:3, col:29> col:14 class S 'int'
// CHECK: EnumDecl {{.*}} <line:[[@LINE-15]]:3, col:18> col:14 class U 'int'

// Instantiated member enums.
// CHECK: ClassTemplateSpecializationDecl {{.*}} struct A definition
// CHECK: EnumDecl {{.*}} <line:[[@LINE-21]]:3, col:25> col:8 E instantiated_from
// CHECK: EnumConstantDecl {{.*}} <col:12, col:16> col:12 X 'A<int>::E'
// CHECK: EnumConstantDecl {{.*}} <col:19, col:23> col:19 Y 'A<int>::E'
// CHECK: EnumDecl {{.*}} <line:[[@LINE-23]]:3, col:29> col:14 referenced class S 'int' instantiated_from
// CHECK: EnumConstantDecl {{.*}} <col:24> col:24 referenced P 'A<int>::S'
// CHECK: EnumConstantDecl {{.*}} <col:27> col:27 Q 'A<int>::S'

// The opaque enum has no brace range.
// CHECK: EnumDecl {{.*}} <line:[[@LINE-27]]:3, col:18> col:14 class U 'int' instantiated_from

// Instantiated local enum.
// CHECK: FunctionDecl {{.*}} used f 'int ()' implicit_instantiation instantiated_from
// CHECK: EnumDecl {{.*}} <col:3, col:26> col:8 L instantiated_from
// CHECK: EnumConstantDecl {{.*}} <col:12, col:24> col:12 referenced M 'L'
