// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fno-diagnostics-show-line-numbers -fcaret-diagnostics-max-lines=1 %s 2>&1 | FileCheck %s -strict-whitespace

auto a() -> int; // ok
const auto b() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'const auto'}}
auto *c() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'auto *'}}
auto (d() -> int); // expected-error {{trailing return type may not be nested within parentheses}}
auto e() -> auto (*)() -> auto (*)() -> void; // ok: same as void (*(*e())())();

namespace GH78694 {

template <typename T> struct B {
  // CHECK:      error: function with trailing return type must specify return type 'auto', not 'void'
  // CHECK-NEXT: {{^}}  template <class U> B(U) -> B<int>;
  // CHECK-NEXT: {{^}}                     ~~~~~~~~^~~~~~{{$}}
  template <class U> B(U) -> B<int>; // expected-error {{function with trailing return type must specify return type 'auto', not 'void'}}
};
}
