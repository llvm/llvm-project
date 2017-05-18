// RUN: %clang_cc1 -x c++-module-map -fmodule-name=A -verify %s -fmodules-local-submodule-visibility
module A { module B {} module C {} }

#pragma clang module contents

#pragma clang module begin A.B
extern "C++" {
  #pragma clang module begin A.C
  template<typename T> void f(T t);
  #pragma clang module end

  void g() { f(0); } // ok
}

extern "C++" {
  #pragma clang module begin A.C
  } // expected-error {{extraneous closing brace}}
  #pragma clang module end
  
  #pragma clang module begin A.C
  extern "C++" { // expected-note {{to match this '{'}}
  #pragma clang module end // expected-error {{expected '}' at end of module}}
}

#pragma clang module end
