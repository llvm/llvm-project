// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  struct MemberClass {
    T member; // expected-error{{with function type}}
  };
  
  T* f0(T* ptr) { 
    return ptr + 1; // expected-error{{pointer to the function}}
  } 
  
  static T* static_member;
};

template<typename T>
T* X0<T>::static_member = ((T*)0) + 1; // expected-error{{pointer to the function}}

template class X0<int>; // okay

template class X0<int(int)>; // expected-note 3{{requested here}}

// Specialize everything, so that the explicit instantiation does not trigger
// any diagnostics.
template<>
struct X0<int(long)>::MemberClass { };

typedef int int_long_func(long);
template<>
int_long_func *X0<int_long_func>::f0(int_long_func *) { return 0; }

template<>
int_long_func *X0<int(long)>::static_member;

template class X0<int(long)>;

