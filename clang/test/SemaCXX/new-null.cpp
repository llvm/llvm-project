// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

typedef __SIZE_TYPE__ size_t;

#if __cplusplus >= 201103L
struct S1 {
   void *operator new(size_t n) {
     return nullptr; // expected-warning {{'operator new' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
   }
   void *operator new[](size_t n) noexcept {
     return __null;
   }
};
#endif

struct S2 {
   static size_t x;
   void *operator new(size_t n) throw() {
     return 0;
   }
   void *operator new[](size_t n) {
     return (void*)0;
#if __cplusplus >= 201103L
     // expected-warning@-2 {{'operator new[]' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
#else
     // expected-warning-re@-4 {{'operator new[]' should not return a null pointer unless it is declared 'throw()'{{$}}}}
#endif
   }
};

struct S3 {
   void *operator new(size_t n) {
     return 1-1;
#if __cplusplus >= 201103L
     // expected-error@-2 {{cannot initialize return object of type 'void *' with an rvalue of type 'int'}}
#else
     // expected-warning@-4 {{expression which evaluates to zero treated as a null pointer constant of type 'void *'}}
     // expected-warning@-5 {{'operator new' should not return a null pointer unless it is declared 'throw()'}}
#endif
   }
   void *operator new[](size_t n) {
     return (void*)(1-1); // expected-warning {{'operator new[]' should not return a null pointer unless it is declared 'throw()'}}
   }
};

#if __cplusplus >= 201103L
template<bool B> struct S4 {
  void *operator new(size_t n) noexcept(B) {
    return 0; // expected-warning {{'operator new' should not return a null pointer}}
  }
};
template struct S4<true>;
template struct S4<false>; // expected-note {{in instantiation of}}
#endif

template<typename ...T> struct S5 { // expected-warning 0-1{{extension}}
  void *operator new(size_t n) throw(T...) {
    return 0; // expected-warning {{'operator new' should not return a null pointer}}
  }
};
template struct S5<>;
template struct S5<int>; // expected-note {{in instantiation of}}
