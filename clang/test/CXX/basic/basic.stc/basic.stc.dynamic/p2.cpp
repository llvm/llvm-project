// RUN: %clang_cc1 -fsyntax-only -fexceptions -fcxx-exceptions -verify -Wno-dynamic-exception-spec %std_cxx98- %s
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

void* operator new(std::size_t) throw(std::bad_alloc);
#if __cplusplus < 201103L
// expected-note@-2 {{previous declaration}}
#endif

void* operator new[](std::size_t) throw(std::bad_alloc); 
void operator delete(void*) throw(); // expected-note{{previous declaration}}
void operator delete[](void*) throw();

void* operator new(std::size_t);
#if __cplusplus < 201103L
// expected-warning@-2 {{'operator new' is missing exception specification 'throw(std::bad_alloc)'}}
#endif
void operator delete(void*);
#if __cplusplus < 201103L
// expected-warning@-2 {{'operator delete' is missing exception specification 'throw()'}}
#else
// expected-warning@-4 {{previously declared with an explicit exception specification redeclared with an implicit}}
#endif
