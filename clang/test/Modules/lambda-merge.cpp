// RUN: %clang_cc1 -fmodules -std=c++17 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fmodules -std=c++17 -emit-llvm %s -o - -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter | FileCheck %s

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<typename T> T f(T v) {
  v();
  return v;
}
inline auto g() {
  int n = 0;
  return f([=] { return n; });
}

template<typename T> constexpr T f2(T v) {
  v();
  return v;
}
constexpr auto g2() {
  int n = 0;
  return f2([=] { return n; });
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
template<typename T> T f(T v) {
  v();
  return v;
}
inline auto g() {
  int n = 0;
  return f([=] { return n; });
}

template<typename T> constexpr T f2(T v) {
  v();
  return v;
}
constexpr auto g2() {
  int n = 0;
  return f2([=] { return n; });
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A
#pragma clang module import B

// CHECK: define {{.*}}use_g
int use_g() {
  return g()();
}

static_assert(g2()() == 0);
