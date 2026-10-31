// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -freflection \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -freflection -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++26 -freflection \
// RUN:            -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -freflection -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++26 -freflection -include-pch %t -ast-print /dev/null | FileCheck %s --check-prefix=PRINT

void TestReflection() {
  constexpr auto x = ^^int;
  // CHECK:  | `-VarDecl {{.*}} x 'const std::meta::info' constexpr cinit
  // CHECK-NEXT:  |   |-value: Reflection ^^int

  constexpr decltype(^^int) y{};
  // CHECK:    `-VarDecl {{.*}} y {{.*}} constexpr listinit
  // CHECK-NEXT:        |-value: Reflection std::meta::info{}
}

struct Record {};
void TestTypeReflection() {
  constexpr auto record = ^^Record;
  // CHECK: VarDecl {{.*}} record 'const std::meta::info' constexpr cinit
  // CHECK-NEXT: value: Reflection ^^Record
  constexpr auto function = ^^void(int);
  // CHECK: VarDecl {{.*}} function 'const std::meta::info' constexpr cinit
  // CHECK-NEXT: value: Reflection ^^void (int)
}

using OrdinaryAlias = int;
template <int N = 2> using AliasTemplate = int;
void TestAliasReflection() {
  constexpr auto ordinary = ^^OrdinaryAlias;
  // CHECK: VarDecl {{.*}} ordinary 'const std::meta::info' constexpr cinit
  // CHECK-NEXT: value: Reflection ^^OrdinaryAlias
  constexpr auto expression = ^^AliasTemplate<1 + 1>;
  // CHECK: VarDecl {{.*}} expression 'const std::meta::info' constexpr cinit
  // CHECK-NEXT: value: Reflection ^^AliasTemplate<2>
  constexpr auto defaulted = ^^AliasTemplate<>;
  // CHECK: VarDecl {{.*}} defaulted 'const std::meta::info' constexpr cinit
  // CHECK-NEXT: value: Reflection ^^AliasTemplate<2>
  // PRINT: expression = ^^AliasTemplate<2>
  // PRINT: defaulted = ^^AliasTemplate<2>
}
