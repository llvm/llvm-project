// RUN: %clang_cc1 -fsyntax-only -verify=enabled,sfinae -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify=sfinae -std=c++20 -Wno-void-ptr-dereference %s

void f(void* p) {
  (void)*p; // enabled-error{{ISO C++ does not allow indirection on operand of type 'void *'}}
}

template<class T>
concept deref = requires (T& t) {
      { *t }; // #FAILED_REQ
};

static_assert(deref<void*>);
// sfinae-error@-1{{static assertion failed}}
// sfinae-note@-2{{because 'void *' does not satisfy 'deref'}}
// sfinae-note@#FAILED_REQ{{because '*t' would be invalid: ISO C++ does not allow indirection on operand of type 'void *'}}
