// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -Wpre-c++17-compat %s
// expected-no-diagnostics

namespace GH57362 {
template <int num>
class TemplateClass {};

template <TemplateClass nttp> // ok, no diagnostic expected
void func() {}
}
