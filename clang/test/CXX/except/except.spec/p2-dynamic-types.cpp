// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -fsyntax-only -Wno-dynamic-exception-spec -verify %s

// Dynamic specifications: valid types.

struct Incomplete; // expected-note 3 {{forward declaration}}

// Exception spec must not have incomplete types, or pointers to them, except
// void.
void ic1() throw(void); // expected-error {{incomplete type 'void' is not allowed in exception specification}}
void ic2() throw(Incomplete); // expected-error {{incomplete type 'Incomplete' is not allowed in exception specification}}
void ic3() throw(void*);
void ic4() throw(Incomplete*); // expected-error {{pointer to incomplete type 'Incomplete' is not allowed in exception specification}}
void ic5() throw(Incomplete&); // expected-error {{reference to incomplete type 'Incomplete' is not allowed in exception specification}}

// Don't suppress errors in template instantiation.
template <typename T> struct TEx; // expected-note {{template is declared here}}

void tf() throw(TEx<int>); // expected-error {{implicit instantiation of undefined template}}

// DR 437, class throws itself.
struct DR437 {
   void f() throw(DR437);
   void g() throw(DR437*);
   void h() throw(DR437&);
};

// DR 437 within a nested class
struct DR437_out {
   struct DR437_in {
      void f() throw(DR437_out);
      void g() throw(DR437_out*);
      void h() throw(DR437_out&);
   }; 
};
