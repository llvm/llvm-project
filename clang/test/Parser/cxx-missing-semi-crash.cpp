// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

// Ensure no crash when DiagnoseMissingSemiAfterTagDefinition encounters an
// annot_template_id not followed by '::'.

namespace GH207992 {
template <typename T> void foo(); // expected-note {{'foo' declared here}}
void bar() {
  union {
  } ::foo<int>; // expected-error {{no template named 'foo' in the global namespace; did you mean simply 'foo'?}} \
                // expected-error {{template specialization requires 'template<>'}}
}
} // namespace N
// expected-error@* {{templates can only be declared in namespace or class scope}}
