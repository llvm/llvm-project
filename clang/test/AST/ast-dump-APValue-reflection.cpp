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

void TestReflection() {
  constexpr auto x = ^^int;
  // CHECK:  | `-VarDecl {{.*}} x 'const std::meta::info' constexpr cinit
  // CHECK-NEXT:  |   |-value: Reflection ^^int

  constexpr decltype(^^int) y{};
  // CHECK:    `-VarDecl {{.*}} y {{.*}} constexpr listinit
  // CHECK-NEXT:        |-value: Reflection std::meta::info{}
}
