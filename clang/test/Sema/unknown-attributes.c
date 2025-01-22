// RUN: %clang_cc1 -fsyntax-only -Wunknown-attribute-namespaces=foo,bar -std=c23 %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-attribute-namespaces=foo,bar -std=c++2b %s 2>&1 | FileCheck %s

[[foo::a(b((c)) d(e((f)))), foo::g(h k)]]
int f1(void) {
  return 0;
}

[[bar::a(b((c)) d(e((f)))), bar::g(h k)]]
int f2(void) {
  return 0;
}

// CHECK: 17:3: warning: unknown attribute namespace 'baz'; attribute 'baz::a' ignored [-Wunknown-attribute-namespaces]
// CHECK: 17:29: warning: unknown attribute namespace 'baz'; attribute 'baz::g' ignored [-Wunknown-attribute-namespaces]

[[baz::a(b((c)) d(e((f)))), baz::g(h k)]]
int f3(void) {
  return 0;
}

// CHECK: 25:3: warning: unknown attribute 'a' ignored [-Wunknown-attributes]
// CHECK: 25:31: warning: unknown attribute 'g' ignored [-Wunknown-attributes]

[[clang::a(b((c)) d(e((f)))), clang::g(h k)]]
int f4(void) {
  return 0;
}

[[clang::noinline]]
int f5(void) {
  return 0;
}
