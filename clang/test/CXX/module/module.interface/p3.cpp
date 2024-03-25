// RUN: %clang_cc1 -std=c++20 %s -verify -pedantic-errors

// As amended by P2615R1 applied as a DR against C++20.
export module p3;

namespace A { int ns_mem; } // expected-note 2{{target}}

// An exported declaration shall declare at least one name.
export; // No diagnostic after P2615R1 DR
export static_assert(true); // No diagnostic after P2615R1 DR
export using namespace A;   // No diagnostic after P2615R1 DR

export { // No diagnostic after P2615R1 DR
  ; // No diagnostic after P2615R1 DR
  static_assert(true); // No diagnostic after P2615R1 DR
  using namespace A;   // No diagnostic after P2615R1 DR
}

export struct {}; // expected-error {{must be class member}} expected-error {{GNU extension}} expected-error {{does not declare anything}}
export struct {} struct_;
export union {}; // expected-error {{must be declared 'static'}} expected-error {{does not declare anything}}
export union {} union_;
export enum {}; // expected-error {{does not declare anything}}
export enum {} enum_;
export enum E : int;
export typedef int; // expected-error {{typedef requires a name}}
export static union {}; // expected-error {{does not declare anything}}
export asm(""); // No diagnostic after P2615R1 DR
export namespace B = A;
export using A::ns_mem; // expected-error {{using declaration referring to 'ns_mem' with module linkage cannot be exported}}
namespace A {
  export using A::ns_mem; // expected-error {{using declaration referring to 'ns_mem' with module linkage cannot be exported}}
}
export using Int = int;
export extern "C++" {} // No diagnostic after P2615R1 DR
export extern "C++" { extern "C" {} } // No diagnostic after P2615R1 DR
export extern "C++" { extern "C" int extern_c; }
export { // No diagnostic after P2615R1 DR
  extern "C++" int extern_cxx;
  extern "C++" {} // No diagnostic after P2615R1 DR
}
export [[]]; // No diagnostic after P2615R1 DR
export [[example::attr]]; // expected-warning {{unknown attribute 'attr'}}

// [...] shall not declare a name with internal linkage
export static int a; // expected-error {{declaration of 'a' with internal linkage cannot be exported}}
export static int b(); // expected-error {{declaration of 'b' with internal linkage cannot be exported}}
export namespace { } // expected-error {{anonymous namespaces cannot be exported}}
export namespace { int c; } // expected-error {{anonymous namespaces cannot be exported}}
namespace { // expected-note {{here}}
  export int d; // expected-error {{export declaration appears within anonymous namespace}}
}
export template<typename> static int e; // expected-error {{declaration of 'e' with internal linkage cannot be exported}}
export template<typename> static int f(); // expected-error {{declaration of 'f' with internal linkage cannot be exported}}
export const int k = 5;
export static union { int n; }; // expected-error {{declaration of 'n' with internal linkage cannot be exported}}
