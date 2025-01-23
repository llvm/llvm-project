// RUN: %clang_cc1 -fsyntax-only -Wunknown-attribute-namespaces=foo,bar -std=c23 %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTR_NS
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-attribute-namespaces=foo,bar -std=c++2b %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTR_NS

// RUN: %clang_cc1 -fsyntax-only -Wunknown-attributes -std=c23 %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTRS
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-attributes -std=c++2b %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTRS

// RUN: %clang_cc1 -fsyntax-only -Wunknown-attributes -Wno-unknown-attribute-namespaces -std=c23 %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTRS_AND_NO_UNKNOWN_ATTR_NS
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-attributes -Wno-unknown-attribute-namespaces -std=c++2b %s 2>&1 | FileCheck %s --check-prefixes=CHECK_UNKNOWN_ATTRS_AND_NO_UNKNOWN_ATTR_NS

// RUN: %clang_cc1 -fsyntax-only -Wno-unknown-attributes -Wunknown-attribute-namespaces -std=c23 %s 2>&1 | FileCheck %s --check-prefixes=CHECK_NO_UNKNOWN_ATTRS_AND_UNKNOWN_ATTR_NS
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unknown-attributes -Wunknown-attribute-namespaces -std=c++2b %s 2>&1 | FileCheck %s --check-prefixes=CHECK_NO_UNKNOWN_ATTRS_AND_UNKNOWN_ATTR_NS

[[foo::a(b((c)) d(e((f)))), foo::g(h k)]]
int f1(void) {
  return 0;
}

[[bar::a(b((c)) d(e((f)))), bar::g(h k)]]
int f2(void) {
  return 0;
}

// CHECK_UNKNOWN_ATTR_NS: 25:3: warning: unknown attribute namespace 'baz'; attribute 'baz::a' ignored [-Wunknown-attribute-namespaces]
// CHECK_UNKNOWN_ATTR_NS: 25:29: warning: unknown attribute namespace 'baz'; attribute 'baz::g' ignored [-Wunknown-attribute-namespaces]
[[baz::a(b((c)) d(e((f)))), baz::g(h k)]]
int f3(void) {
  return 0;
}

// CHECK_UNKNOWN_ATTR_NS: 32:3: warning: unknown attribute 'a' ignored [-Wunknown-attributes]
// CHECK_UNKNOWN_ATTR_NS: 32:31: warning: unknown attribute 'g' ignored [-Wunknown-attributes]
[[clang::a(b((c)) d(e((f)))), clang::g(h k)]]
int f4(void) {
  return 0;
}

[[clang::noinline]]
int f5(void) {
  return 0;
}

// CHECK_UNKNOWN_ATTRS: 44:20: warning: unknown attribute 'b' ignored [-Wunknown-attributes]
// CHECK_UNKNOWN_ATTRS: 44:3: warning: unknown attribute namespace 'unknown_ns'; attribute 'unknown_ns::a' ignored [-Wunknown-attribute-namespaces]
[[unknown_ns::a]][[gnu::b]]
int f6(void) {
  return 0;
}

// CHECK_UNKNOWN_ATTRS_AND_NO_UNKNOWN_ATTR_NS: 50:20: warning: unknown attribute 'b' ignored [-Wunknown-attributes]
[[unknown_ns::a]][[gnu::b]]
int f7(void) {
  return 0;
}

// CHECK_NO_UNKNOWN_ATTRS_AND_UNKNOWN_ATTR_NS: 56:3: warning: unknown attribute namespace 'unknown_ns'; attribute 'unknown_ns::a' ignored [-Wunknown-attribute-namespaces]
[[unknown_ns::a]][[gnu::b]]
int f8(void) {
  return 0;
}
