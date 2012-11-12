// RUN: %clang_cc1 -fsyntax-only -verify -Wno-bool-conversion %s

typedef __typeof__(((int*)0)-((int*)0)) ptrdiff_t;

namespace DontResolveTooEarly_WaitForOverloadResolution
{
  template <class T> T* f(int);	// #1 
  template <class T, class U> T& f(U); // #2 
  
  void g() {
    int *ip = f<int>(1);	// calls #1
  }

  template <class T> 
    T* f2(int); 
  template <class T, class U> 
    T& f2(U); 

  void g2() {
    int*ip = (f2<int>)(1); // ok
  }

} // End namespace

namespace DontAllowUnresolvedOverloadedExpressionInAnUnusedExpression
{
  void one() { }
  template<class T> void oneT() { }

  void two() { } // expected-note 2 {{possible target for call}}
  void two(int) { } // expected-note 2 {{possible target for call}}
  template<class T> void twoT() { }  // expected-note 2 {{possible target for call}}
  template<class T> void twoT(T) { }  // expected-note 2 {{possible target for call}}

  void check()
  {
    one; // expected-warning {{expression result unused}}
    two; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
    oneT<int>; // expected-warning {{expression result unused}}
    twoT<int>; // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
  }

  // check the template function case
  template<class T> void check()
  {
    one; // expected-warning {{expression result unused}}
    two; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
    oneT<int>; // expected-warning {{expression result unused}}
    twoT<int>; // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
 
  }

}

  template<typename T>
    void twoT() { }
  template<typename T, typename U>
    void twoT(T) { }
  

  void two() { }; //expected-note 5{{candidate}}
  void two(int) { }; //expected-note 5{{candidate}}
 


  void one() { }
  template<class T> 
    void oneT() { }

  template<class T>
  void cant_resolve() { } //expected-note 3{{candidate}}
 
  template<class T> void cant_resolve(T) { }//expected-note 3{{candidate}}
 

int main()
{
  
  { static_cast<void>(one); }
  { (void)(one); }
  { static_cast<void>(oneT<int>); }
  { (void)(oneT<int>); }

  { static_cast<void>(two); } // expected-error {{address of overloaded function 'two' cannot be static_cast to type 'void'}}
  { (void)(two); } // expected-error {{address of overloaded function 'two' cannot be cast to type 'void'}}
  { static_cast<void>(twoT<int>); } 
  { (void)(twoT<int>); } 


  { ptrdiff_t x = reinterpret_cast<ptrdiff_t>(oneT<int>); } 
  { (void) reinterpret_cast<int (*)(char, double)>(oneT<int>); } 
  { (void) reinterpret_cast<ptrdiff_t>(one); }
  { (void) reinterpret_cast<int (*)(char, double)>(one); }

  { ptrdiff_t x = reinterpret_cast<ptrdiff_t>(twoT<int>); }  
  { (void) reinterpret_cast<int (*)(char, double)>(twoT<int>); } 
  { (void) reinterpret_cast<void (*)(int)>(two); } //expected-error {{reinterpret_cast}}
  { (void) static_cast<void (*)(int)>(two); } //ok

  { (void) reinterpret_cast<int>(two); } //expected-error {{reinterpret_cast}}
  { (void) reinterpret_cast<int (*)(char, double)>(two); } //expected-error {{reinterpret_cast}}

  { bool b = (twoT<int>); } 
  { bool b = (twoT<int, int>); } 

  { bool b = &twoT<int>; //&foo<int>; }
    b = &(twoT<int>); }

  { ptrdiff_t x = (ptrdiff_t) &twoT<int>;
      x = (ptrdiff_t) &twoT<int>; }

  { ptrdiff_t x = (ptrdiff_t) twoT<int>;
      x = (ptrdiff_t) twoT<int>; }

  
  { ptrdiff_t x = (ptrdiff_t) &twoT<int,int>;
  x = (ptrdiff_t) &twoT<int>; }

  { oneT<int>;   &oneT<int>; } //expected-warning 2{{expression result unused}}
  { static_cast<void>(cant_resolve<int>); } // expected-error {{address of overload}}
  { bool b = cant_resolve<int>; } // expected-error {{address of overload}}
  { (void) cant_resolve<int>; } // expected-error {{address of overload}}

}

namespace member_pointers {
  struct S {
    template <typename T> bool f(T) { return false; }
    template <typename T> static bool g(T) { return false; }

    template <typename T> bool h(T) { return false; }  // expected-note 3 {{possible target for call}}
    template <int N> static bool h(int) { return false; } // expected-note 3 {{possible target for call}}
  };

  void test(S s) {
    if (S::f<char>) return; // expected-error {{call to non-static member function without an object argument}}
    if (S::f<int>) return; // expected-error {{call to non-static member function without an object argument}}
    if (&S::f<char>) return;
    if (&S::f<int>) return;
    if (s.f<char>) return; // expected-error {{reference to non-static member function must be called}}
    if (s.f<int>) return; // expected-error {{reference to non-static member function must be called}}
    if (&s.f<char>) return; // expected-error {{cannot create a non-constant pointer to member function}}
    if (&s.f<int>) return; // expected-error {{cannot create a non-constant pointer to member function}}

    if (S::g<char>) return;
    if (S::g<int>) return;
    if (&S::g<char>) return;
    if (&S::g<int>) return;
    if (s.g<char>) return;
    if (s.g<int>) return;
    if (&s.g<char>) return;
    if (&s.g<int>) return;

    if (S::h<42>) return;
    if (S::h<int>) return; // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
    if (&S::h<42>) return;
    if (&S::h<int>) return;
    if (s.h<42>) return;
    if (s.h<int>) return; // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
    if (&s.h<42>) return;
    if (&s.h<int>) return; // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}

    { bool b = S::f<char>; } // expected-error {{call to non-static member function without an object argument}}
    { bool b = S::f<int>; } // expected-error {{call to non-static member function without an object argument}}
    { bool b = &S::f<char>; }
    { bool b = &S::f<int>; }
    // These next two errors are terrible.
    { bool b = s.f<char>; } // expected-error {{reference to non-static member function must be called}}
    { bool b = s.f<int>; } // expected-error {{reference to non-static member function must be called}}
    { bool b = &s.f<char>; } // expected-error {{cannot create a non-constant pointer to member function}}
    { bool b = &s.f<int>; } // expected-error {{cannot create a non-constant pointer to member function}}

    { bool b = S::g<char>; }
    { bool b = S::g<int>; }
    { bool b = &S::g<char>; }
    { bool b = &S::g<int>; }
    { bool b = s.g<char>; }
    { bool b = s.g<int>; }
    { bool b = &s.g<char>; }
    { bool b = &s.g<int>; }

    { bool b = S::h<42>; }
    { bool b = S::h<int>; } // expected-error {{can't form member pointer of type 'bool' without '&' and class name}}
    { bool b = &S::h<42>; }
    { bool b = &S::h<int>; }
    { bool b = s.h<42>; }
    { bool b = s.h<int>; } // expected-error {{can't form member pointer of type 'bool' without '&' and class name}}
    { bool b = &s.h<42>; }
    { bool b = &s.h<int>; } // expected-error {{can't form member pointer of type 'bool' without '&' and class name}}
  }
}
