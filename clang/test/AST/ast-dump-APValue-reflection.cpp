// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -freflection \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace %s

void TestReflection() {
  constexpr auto x = ^^int;
  // CHECK:  | `-VarDecl {{.*}} x 'const std::meta::info' constexpr cinit
  // CHECK-NEXT:  |   |-value: Reflection ^^int

  constexpr decltype(^^int) y{};
  // CHECK:    `-VarDecl {{.*}} y {{.*}} constexpr listinit
  // CHECK-NEXT:        |-value: Reflection std::meta::info{}
}
