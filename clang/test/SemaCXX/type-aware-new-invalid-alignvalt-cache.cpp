// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -std=c++26 \
// RUN: -fno-aligned-allocation -Wno-ext-cxx-type-aware-allocators -verify %s

void first() {
  new int;
}

namespace std {
  using size_t = __SIZE_TYPE__;
  template <class T> struct type_identity { using type = T; };
}

void second() {
  new float;
}

namespace std {
  enum class align_val_t : size_t {};
}

template <class T> void *operator new(std::type_identity<T>, std::size_t, std::align_val_t) = delete; // #new_decl
template <class T> void operator delete(std::type_identity<T>, void *, std::size_t, std::align_val_t) = delete;

struct Foo {
  int x; 
};

void third() {
  (void)new Foo; // #new_expr
  // expected-error@#new_expr {{call to deleted function 'operator new'}}
  // expected-note@#new_decl {{candidate function [with T = Foo] has been explicitly deleted}}
}
