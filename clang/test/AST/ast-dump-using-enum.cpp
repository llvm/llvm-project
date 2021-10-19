// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump -ast-dump-filter Foo %s | FileCheck -strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-unknown -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Foo /dev/null \
// RUN: | FileCheck --strict-whitespace %s

namespace Bob {
enum class Foo {
  Foo_a,
  Foo_b
};
}; // namespace Bob

using enum Bob::Foo;

// CHECK-LABEL: Dumping Bob::Foo
// CHECK-NEXT: EnumDecl {{.*}} class Foo 'int'
// CHECK-NEXT: |-EnumConstantDecl {{.*}} Foo_a 'Bob::Foo'
// CHECK-NEXT: `-EnumConstantDecl {{.*}} Foo_b 'Bob::Foo'

// CHECK-LABEL: Dumping Foo:
// CHECK-NEXT: UsingEnumDecl {{.*}} Enum {{.*}} 'Foo'

// CHECK-LABEL: Dumping Foo_a:
// CHECK-NEXT: UsingShadowDecl {{.*}} implicit EnumConstant {{.*}} 'Foo_a' 'Bob::Foo'

// CHECK-LABEL: Dumping Foo_b:
// CHECK-NEXT: UsingShadowDecl {{.*}} implicit EnumConstant {{.*}} 'Foo_b' 'Bob::Foo'
