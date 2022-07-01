// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -std=c++2a -x c++-header %S/Inputs/header.h -emit-header-module -fmodule-name=FIXME -o %t/h.pcm
// RUN: %clang_cc1 -std=c++2a %s -DX_INTERFACE -emit-module-interface -o %t/x.pcm
// RUN: %clang_cc1 -std=c++2a %s -DY_INTERFACE -emit-module-interface -o %t/y.pcm
// RUN: %clang_cc1 -std=c++2a %s -DINTERFACE -fmodule-file=%t/x.pcm -fmodule-file=%t/y.pcm -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++2a %s -DIMPLEMENTATION -I%S/Inputs -fmodule-file=%t/h.pcm -fmodule-file=%t/m.pcm -verify
// RUN: %clang_cc1 -std=c++2a %s -DUSER -I%S/Inputs -fmodule-file=%t/h.pcm -fmodule-file=%t/m.pcm -verify

#if defined(X_INTERFACE)
export module X;
export int x;

#elif defined(Y_INTERFACE)
export module Y;
export int y;

#elif defined(INTERFACE)
export module p2;
export import X;
import Y; // not exported

namespace A {
  int f();
  export int g();
  int h();
  namespace inner {}
}
export namespace B {
  namespace inner {}
}
namespace B {
  int f();
}
namespace C {}
namespace D { int f(); }
export namespace D {}

#elif defined(IMPLEMENTATION)
module p2;
import "header.h";

// Per [basic.scope.namespace]/2.3, exportedness has no impact on visibility
// within the same module.
//
// expected-no-diagnostics

void use() {
  A::f();
  A::g();
  A::h();
  using namespace A::inner;

  using namespace B;
  using namespace B::inner;
  B::f();
  f();

  using namespace C;

  D::f();
}

int use_header() { return foo + bar::baz(); }

#elif defined(USER)
import p2;
import "header.h";

void use() {
  // namespace A is implicitly exported by the export of A::g.
  A::f(); // expected-error {{no member named 'f' in namespace 'A'}}
  A::g();
  A::h(); // expected-error {{no member named 'h' in namespace 'A'}}
  using namespace A::inner; // expected-error {{expected namespace name}}

  // namespace B and B::inner are explicitly exported
  using namespace B;
  using namespace B::inner;
  B::f(); // expected-error {{no member named 'f' in namespace 'B'}}
  f(); // expected-error {{undeclared identifier 'f'}}

  // namespace C is not exported
  using namespace C; // expected-error {{expected namespace name}}

  // namespace D is exported, but D::f is not
  D::f(); // expected-error {{no member named 'f' in namespace 'D'}}
}

int use_header() { return foo + bar::baz(); }

#else
#error unknown mode
#endif
