// RUN: %clang_cc1 -fsyntax-only -fexceptions -fcxx-exceptions -verify %s
int *use_new(int N) {
  if (N == 1)
    return new int;
  
  return new int [N];
}

void use_delete(int* ip, int N) {
  if (N == 1)
    delete ip;
  else
    delete [] ip;
}

namespace std {
  class bad_alloc { };
  
  typedef __SIZE_TYPE__ size_t;
}

void* operator new(std::size_t) throw(std::bad_alloc); // expected-note{{previous declaration}}
void* operator new[](std::size_t) throw(std::bad_alloc); 
void operator delete(void*) throw(); // expected-note{{previous declaration}}
void operator delete[](void*) throw();

void* operator new(std::size_t); // expected-warning{{'operator new' is missing exception specification 'throw(std::bad_alloc)'}}
void operator delete(void*); // expected-warning{{'operator delete' is missing exception specification 'throw()'}}
