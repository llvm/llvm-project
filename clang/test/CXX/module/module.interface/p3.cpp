// RUN: %clang_cc1 -std=c++2a %s -verify -pedantic-errors

export module p3;

namespace A { int ns_mem; }

// An exported declaration shall declare at least one name.
export; // expected-error {{empty declaration cannot be exported}}
export static_assert(true); // expected-error {{static_assert declaration cannot be exported}}
export using namespace A; // expected-error {{ISO C++20 does not permit using directive to be exported}}

export { // expected-note 3{{export block begins here}}
  ; // expected-error {{ISO C++20 does not permit an empty declaration to appear in an export block}}
  static_assert(true); // expected-error {{ISO C++20 does not permit a static_assert declaration to appear in an export block}}
  using namespace A; // expected-error {{ISO C++20 does not permit using directive to be exported}}
}

export struct {}; // expected-error {{must be class member}} expected-error {{GNU extension}}
export struct {} struct_;
export union {}; // expected-error {{must be declared 'static'}}
export union {} union_;
export enum {}; // expected-error {{does not declare anything}}
export enum {} enum_;
export enum E : int;
export typedef int; // expected-error {{typedef requires a name}}
export static union {}; // FIXME: this declaration is ill-formed even without the 'export'
export asm(""); // expected-error {{asm declaration cannot be exported}}
export namespace B = A;
export using A::ns_mem;
namespace A {
  export using A::ns_mem;
}
export using Int = int;
export extern "C++" {} // expected-error {{ISO C++20 does not permit a declaration that does not introduce any names to be exported}}
export extern "C++" { extern "C" {} } // expected-error {{ISO C++20 does not permit a declaration that does not introduce any names to be exported}}
export extern "C++" { extern "C" int extern_c; }
export { // expected-note {{export block}}
  extern "C++" int extern_cxx;
  extern "C++" {} // expected-error {{ISO C++20 does not permit a declaration that does not introduce any names to be exported}}
}
export [[]]; // FIXME (bad diagnostic text): expected-error {{empty declaration cannot be exported}}
export [[example::attr]]; // FIXME: expected-error {{empty declaration cannot be exported}} expected-warning {{unknown attribute 'attr'}}

// [...] shall not declare a name with internal linkage
export static int a; // expected-error {{declaration of 'a' with internal linkage cannot be exported}}
export static int b(); // expected-error {{declaration of 'b' with internal linkage cannot be exported}}
export namespace { int c; } // expected-error {{declaration of 'c' with internal linkage cannot be exported}}
namespace { // expected-note {{here}}
  export int d; // expected-error {{export declaration appears within anonymous namespace}}
}
export template<typename> static int e; // FIXME
export template<typename> static int f(); // expected-error {{declaration of 'f' with internal linkage cannot be exported}}
export const int k = 5;
export static union { int n; }; // expected-error {{declaration of 'n' with internal linkage cannot be exported}}
