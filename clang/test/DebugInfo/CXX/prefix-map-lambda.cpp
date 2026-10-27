// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   -fdebug-prefix-map=%S=/SOURCE_ROOT %s -emit-llvm -o - | FileCheck %s

template <typename T> void b(T) {}
void c() {
  // CHECK: !DISubprogram(name: "b<(lambda at
  // CHECK-SAME:          SOURCE_ROOT
  // CHECK-SAME:          [[@LINE+1]]:{{[0-9]+}})>"
  b([]{});
}

// The lambda in a declaration template argument (here a pointer to member)
// must be remapped as well.
template <typename T> struct S { void m() {} };
template <auto P> void d() {}
void e() {
  auto l = [] {};
  // CHECK: !DISubprogram(name: "d<&S<(lambda at
  // CHECK-SAME:          SOURCE_ROOT
  // CHECK-SAME:          [[@LINE-3]]:{{[0-9]+}})>::m>"
  d<&S<decltype(l)>::m>();
}
