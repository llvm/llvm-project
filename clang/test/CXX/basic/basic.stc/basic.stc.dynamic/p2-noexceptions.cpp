// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
namespace std {
  class bad_alloc { };
  
  typedef __SIZE_TYPE__ size_t;
}

class foo { virtual ~foo(); };

void* operator new(std::size_t); 
void* operator new[](std::size_t);
void operator delete(void*);
void operator delete[](void*);
