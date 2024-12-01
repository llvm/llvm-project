// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp14 -std=gnu++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp14 -std=gnu++14 %s -fdelayed-template-parsing -DDELAYED_TEMPLATE_PARSING
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp17 -std=gnu++1z %s



// Errors
export class foo { };   // expected-error {{expected template}}
template  x;            // expected-error {{a type specifier is required for all declarations}} \
                        // expected-error {{does not refer}}
export template x;      // expected-error {{expected '<' after 'template'}}
export template<class T> class x0; // expected-warning {{exported templates are unsupported}}
template < ;            // expected-error {{expected template parameter}} \
// expected-error{{expected ',' or '>' in template-parameter-list}} \
// expected-error {{declaration does not declare anything}}
template <int +> struct x1; // expected-error {{expected ',' or '>' in template-parameter-list}}

// verifies that we only walk to the ',' & still produce errors on the rest of the template parameters
template <int +, T> struct x2; // expected-error {{expected ',' or '>' in template-parameter-list}} \
                                expected-error {{expected unqualified-id}}
template<template<int+>> struct x3; // expected-error {{expected ',' or '>' in template-parameter-list}} \
                                         cpp14-error {{template template parameter requires 'class' after the parameter list}} \
                                         cpp17-error {{template template parameter requires 'class' or 'typename' after the parameter list}}
template <template X> struct Err1; // expected-error {{expected '<' after 'template'}} \
// expected-error{{extraneous}}
template <template <typename> > struct Err2;       // cpp14-error {{template template parameter requires 'class' after the parameter list}}
// cpp17-error@-1{{template template parameter requires 'class' or 'typename' after the parameter list}}
template <template <typename> Foo> struct Err3;    // cpp14-error {{template template parameter requires 'class' after the parameter list}}
// cpp17-error@-1{{template template parameter requires 'class' or 'typename' after the parameter list}}

template <template <typename> typename Foo> struct Cxx1z;
#if __cplusplus <= 201402L
// expected-warning@-2 {{extension}}
#endif

// Template function declarations
template <typename T> void foo();
template <typename T, typename U> void foo();

// Template function definitions.
template <typename T> void foo() { }

// Template class (forward) declarations
template <typename T> struct A;
template <typename T, typename U> struct b;
template <typename> struct C;
template <typename, typename> struct D;

// Forward declarations with default parameters?
template <typename T = int> class X1;
template <typename = int> class X2;

// Forward declarations w/template template parameters
template <template <typename> class T> class TTP1;
template <template <typename> class> class TTP2;
template <template <typename> class T = foo> class TTP3; // expected-error{{must be a class template}}
template <template <typename> class = foo> class TTP3; // expected-error{{must be a class template}}
template <template <typename X, typename Y> class T> class TTP5;

// Forward declarations with non-type params
template <int> class NTP0;
template <int N> class NTP1;
template <int N = 5> class NTP2;
template <int = 10> class NTP3;
template <unsigned int N = 12u> class NTP4;
template <unsigned int = 12u> class NTP5;
template <unsigned = 15u> class NTP6;
template <typename T, T Obj> class NTP7;

// Template class declarations
template <typename T> struct A { };
template <typename T, typename U> struct B { };

// Template parameter shadowing
template<typename T, // expected-note{{template parameter is declared here}}
         typename T> // expected-error{{declaration of 'T' shadows template parameter}}
  void shadow1();

template<typename T> // expected-note{{template parameter is declared here}}
void shadow2(int T); // expected-error{{declaration of 'T' shadows template parameter}}

template<typename T> // expected-note{{template parameter is declared here}}
class T { // expected-error{{declaration of 'T' shadows template parameter}}
};

template<int Size> // expected-note{{template parameter is declared here}}
void shadow3(int Size); // expected-error{{declaration of 'Size' shadows template parameter}}

template<typename T> // expected-note{{here}}
struct shadow4 {
  int T; // expected-error{{shadows}}
};

template<typename T> // expected-note{{here}}
struct shadow5 {
  int T(int, float); // expected-error{{shadows}}
};

template<typename T, // expected-note{{template parameter is declared here}}
         T T> // expected-error{{declaration of 'T' shadows template parameter}}
void shadow6();

template<typename T, // expected-note{{template parameter is declared here}}
         template<typename> class T> // expected-error{{declaration of 'T' shadows template parameter}}
void shadow7();

// PR8302
template<template<typename> class T> struct shadow8 { // expected-note{{template parameter is declared here}}
  template<template<typename> class T> struct inner; // expected-error{{declaration of 'T' shadows template parameter}}
};

// Non-type template parameters in scope
template<int Size>
void f(int& i) {
  i = Size;
 #ifdef DELAYED_TEMPLATE_PARSING
  Size = i;
 #else
  Size = i; // expected-error{{expression is not assignable}}
 #endif
}

template<typename T>
const T& min(const T&, const T&);

void f2() {
  int x;
  A< typeof(x>1) > a;
}


// PR3844
template <> struct S<int> { }; // expected-error{{explicit specialization of undeclared template struct 'S'}}
template <> union U<int> { }; // expected-error{{explicit specialization of undeclared template union 'U'}}

struct SS;
union UU;
template <> struct SS<int> { }; // expected-error{{explicit specialization of non-template struct 'SS'}}
template <> union UU<int> { }; // expected-error{{explicit specialization of non-template union 'UU'}}

namespace PR6184 {
  namespace N {
    template <typename T>
    void bar(typename T::x);
  }

  template <typename T>
  void N::bar(typename T::x) { }
}

// This PR occurred only in template parsing mode.
namespace PR17637 {
template <int>
struct L {
  template <typename T>
  struct O {
    template <typename U>
    static void Fun(U);
  };
};

template <int k>
template <typename T>
template <typename U>
void L<k>::O<T>::Fun(U) {}

void Instantiate() { L<0>::O<int>::Fun(0); }

}

namespace explicit_partial_specializations {
typedef char (&oneT)[1];
typedef char (&twoT)[2];
typedef char (&threeT)[3];
typedef char (&fourT)[4];
typedef char (&fiveT)[5];
typedef char (&sixT)[6];

char one[1];
char two[2];
char three[3];
char four[4];
char five[5];
char six[6];

template<bool b> struct bool_ { typedef int type; };
template<> struct bool_<false> {  };

#define XCAT(x,y) x ## y
#define CAT(x,y) XCAT(x,y)
#define sassert(_b_) bool_<(_b_)>::type CAT(var, __LINE__);


template <int>
struct L {
  template <typename T>
  struct O {
    template <typename U>
    static oneT Fun(U);

  };
};
template <int k>
template <typename T>
template <typename U>
oneT L<k>::O<T>::Fun(U) { return one; }

template<>
template<>
template<typename U>
oneT L<0>::O<char>::Fun(U) { return one; }


void Instantiate() {
  sassert(sizeof(L<0>::O<int>::Fun(0)) == sizeof(one));
  sassert(sizeof(L<0>::O<char>::Fun(0)) == sizeof(one));
}

}

namespace func_tmpl_spec_def_in_func {
// We failed to diagnose function template specialization definitions inside
// functions during recovery previously.
template <class> void FuncTemplate() {}
void TopLevelFunc() {
  // expected-error@+2 {{expected a qualified name after 'typename'}}
  // expected-error@+1 {{function definition is not allowed here}}
  typename template <> void FuncTemplate<void>() { }
  // expected-error@+1 {{function definition is not allowed here}}
  void NonTemplateInner() { }
}
}

namespace broken_baseclause {
template<typename T>
struct base { };

struct t1 : base<int, // expected-note {{to match this '<'}}
  public:  // expected-error {{expected expression}} expected-error {{expected '>'}}
};
// expected-error@-1 {{expected '{' after base class list}}
struct t2 : base<int, // expected-note {{to match this '<'}}
  public  // expected-error {{expected expression}} expected-error {{expected '>'}}
};
// expected-error@-1 {{expected '{' after base class list}}

}

namespace class_scope_instantiation {
  struct A {
    template<typename T> void f(T);
    template void f<int>(int); // expected-error {{expected '<' after 'template'}}
    template void f(float); // expected-error {{expected '<' after 'template'}}
    extern template // expected-error {{expected member name or ';'}}
      void f(double);
  };
}

namespace PR42071 {
  template<int SomeTemplateName<void>> struct A; // expected-error {{parameter name cannot have template arguments}}
  template<int operator+> struct B; // expected-error {{'operator+' cannot be the name of a parameter}}
  struct Q {};
  template<int Q::N> struct C; // expected-error {{parameter declarator cannot be qualified}}
  template<int f(int a = 0)> struct D; // expected-error {{default arguments can only be specified for parameters in a function declaration}}
}

namespace AnnotateAfterInvalidTemplateId {
  template<int I, int J> struct A { };
  template<int J> struct A<0, J> { }; // expected-note {{J = 0}}
  template<int I> struct A<I, 0> { }; // expected-note {{I = 0}}

  void f() { A<0, 0>::f(); } // expected-error {{ambiguous partial specializations}}
}

namespace PR45063 {
  template<class=class a::template b<>> struct X {}; // expected-error {{undeclared identifier 'a'}}
}

namespace NoCrashOnEmptyNestedNameSpecifier {
  template <typename FnT,
            typename T = typename ABC<FnT>::template arg_t<0>> // expected-error {{no template named 'ABC'}}
  void foo(FnT) {}
}

namespace PR45239 {
  // Ensure we don't crash here. We used to deallocate the TemplateIdAnnotation
  // before we'd parsed it.
  template<int> int b;
  template<int> auto f() -> b<0>; // expected-error +{{}}
}

namespace PR46231 {
  template; // expected-error {{declaration does not declare anything}}
  template<>; // expected-error {{declaration does not declare anything}}
  template<int>; // expected-error {{declaration does not declare anything}}
  template int; // expected-error {{declaration does not declare anything}}
  template<> int; // expected-error {{declaration does not declare anything}}
  template<int> int; // expected-error {{declaration does not declare anything}}
}

namespace PR99933 {
  void foo() { template <typename> int i; } // expected-error {{templates can only be declared in namespace or class scope}}
}
