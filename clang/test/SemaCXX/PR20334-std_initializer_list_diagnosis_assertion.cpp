// RUN: %clang_cc1 -std=c++11 -verify=cxx11 -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify=cxx98 %s
// RUN: %clang_cc1 -std=c++11 -verify=cxx11 -emit-llvm-only %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify=cxx98 %s -fexperimental-new-constant-interpreter


namespace std {
  template <class _E>
  class initializer_list {};
  // cxx11-error@-1 {{'std::initializer_list<int>' layout not recognized. Must be a non-polymorphic class type with no bases and two fields: a 'const E *' and either another 'const E *' or a 'std::size_t'}}
}

template<class E> int f(std::initializer_list<E> il);
	

int F = f({1, 2, 3});
// cxx98-error@-1 {{expected expression}}
