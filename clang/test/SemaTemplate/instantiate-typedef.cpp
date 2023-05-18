// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct add_pointer {
  typedef T* type; // expected-error{{'type' declared as a pointer to a reference}}
};

add_pointer<int>::type test1(int * ptr) { return ptr; }

add_pointer<float>::type test2(int * ptr) { 
  return ptr; // expected-error{{cannot initialize return object of type 'add_pointer<float>::type' (aka 'float *') with an lvalue of type 'int *'}}
}

add_pointer<int&>::type // expected-note{{in instantiation of template class 'add_pointer<int &>' requested here}}
test3(); 
