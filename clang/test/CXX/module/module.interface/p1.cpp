// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++2a %t/errors.cpp -verify
// RUN: %clang_cc1 -std=c++2a %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++2a %t/impl.cpp -fmodule-file=M=%t/M.pcm -verify
// RUN: %clang_cc1 -std=c++2a %t/interface-partition.cppm -emit-module-interface -o %t/L-P1.pcm
// RUN: %clang_cc1 -std=c++2a %t/implementation-partition.cpp -fmodule-file=L:P1=%t/L-P1.pcm -verify

//--- errors.cpp
module;
export int a; // expected-error {{export declaration can only be used within a module interface}}
export module M;
export int b; // #1
namespace N {
  export int c; // #2
}

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

//--- M.cppm
export module M;
export int b;
namespace N {
  export int c;
}

//--- impl.cpp
module M; // #M

export int b2; // expected-error {{export declaration can only be used within a module interface}}
namespace N {
  export int c2; // expected-error {{export declaration can only be used within a module interface}}
}
// expected-note@#M 2+{{add 'export'}}

//--- interface-partition.cppm

export module L:P1;

//--- implementation-partition.cpp

module L:P2; // #M

export import :P1; // expected-error {{export declaration can only be used within a module interface}}

export int b3; // expected-error {{export declaration can only be used within a module interface}}
namespace N {
  export int c3; // expected-error {{export declaration can only be used within a module interface}}
}
// expected-note@#M 2+{{add 'export'}}
