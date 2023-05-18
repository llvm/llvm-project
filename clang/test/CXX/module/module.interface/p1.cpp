// RUN: %clang_cc1 -std=c++2a %s -DERRORS -verify
// RUN: %clang_cc1 -std=c++2a %s -emit-module-interface -o %t.pcm
// RUN: %clang_cc1 -std=c++2a %s -fmodule-file=M=%t.pcm -DIMPLEMENTATION -verify -Db=b2 -Dc=c2

module;

#ifdef ERRORS
export int a; // expected-error {{export declaration can only be used within a module purview}}
#endif

#ifndef IMPLEMENTATION
export
#else
// expected-error@#1 {{export declaration can only be used within a module purview}}
// expected-error@#2 {{export declaration can only be used within a module purview}}
// expected-note@+2 1+{{add 'export'}}
#endif
module M;

export int b; // #1
namespace N {
  export int c; // #2
}

#ifdef ERRORS
namespace { // expected-note 2{{anonymous namespace begins here}}
  export int d1; // expected-error {{export declaration appears within anonymous namespace}}
  namespace X {
    export int d2; // expected-error {{export declaration appears within anonymous namespace}}
  }
}

export export int e; // expected-error {{within another export declaration}}
export { export int f; } // expected-error {{within another export declaration}} expected-note {{export block begins here}}

module :private; // expected-note {{private module fragment begins here}}
export int priv; // expected-error {{export declaration cannot be used in a private module fragment}}
#endif
