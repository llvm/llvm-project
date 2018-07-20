// RUN: %clang_cc1 -std=c++98 -verify -fsyntax-only -Wno-c++11-extensions -Wno-c++1y-extensions %s -DPRECXX11
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -Wno-c++1y-extensions %s
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only %s

#ifdef PRECXX11
  #define CONST const
#else
  #define CONST constexpr
#endif

template<typename T> 
T pi = T(3.1415926535897932385); // expected-note 2{{declared here}}

template<typename T> 
CONST T cpi = T(3.1415926535897932385); // expected-note {{template is declared here}}

template<typename T> extern CONST T vc;
#ifndef PRECXX11
// expected-error@-2 {{constexpr variable declaration must be a definition}}
#endif

namespace use_in_top_level_funcs {

  void good() {
    int ipi = pi<int>;
    int icpi = cpi<int>;
    double dpi = pi<double>;
    double dcpi = cpi<double>;
  }

  void no_deduce() {
    // template arguments are not deduced for uses of variable templates.
    int ipi = pi; // expected-error {{use of variable template 'pi' requires template arguments}}
    int icpi = cpi; // expected-error {{use of variable template 'cpi' requires template arguments}}
  }
  
  template<typename T>
  T circular_area(T r) {
    return pi<T> * r * r;
  }

  template<typename T>
  CONST T const_circular_area(T r) {
    return cpi<T> * r * r;
  }

  double use_circular_area(double r) {
    CONST float t = const_circular_area(2.0) - 12;
#ifndef PRECXX11
    static_assert(const_circular_area(2) == 12, "");
    CONST int test = (t > 0) && (t < 1);
    static_assert(test, "");
#endif
    return circular_area(r);
  }
}

namespace shadow {
  void foo() {
    int ipi0 = pi<int>;
    int pi; // expected-note {{found}}
    int a = pi;
    int ipi = pi<int>;  // expected-error {{'pi' does not name a template but is followed by template arguments; did you mean '::pi'?}}
  }
}

namespace odr_tmpl {
  namespace pv_cvt {
    int v;   // expected-note {{previous definition is here}}
    template<typename T> T v; // expected-error {{redefinition of 'v' as different kind of symbol}}
  }
  namespace pvt_cv {
    template<typename T> T v; // expected-note {{previous definition is here}}
    int v;   // expected-error {{redefinition of 'v' as different kind of symbol}}
  }
  namespace pvt_cvt {
    template<typename T> T v0; // expected-note {{previous definition is here}}
    template<typename T> T v0; // expected-error {{redefinition of 'v0'}}

    template<typename T> T v; // expected-note {{previous definition is here}}
    template<typename T> int v; // expected-error {{redefinition of 'v'}}
    
    template<typename T> extern int v1; // expected-note {{previous template declaration is here}}
    template<int I> int v1;      // expected-error {{template parameter has a different kind in template redeclaration}}
  }
  namespace pvt_use {
    template<typename T> T v;
    v = 10;  // expected-error {{C++ requires a type specifier for all declarations}}
  }

  namespace pvt_diff_params {
    template<typename T, typename> T v;   // expected-note 2{{previous template declaration is here}}
    template<typename T> T v;   // expected-error {{too few template parameters in template redeclaration}}
    template<typename T, typename, typename> T v; // expected-error {{too many template parameters in template redeclaration}}
  }

  namespace pvt_extern {
    template<typename T> T v = T();
    template<typename T> extern T v;      // redeclaration is allowed \
                                          // expected-note {{previous declaration is here}}
    template<typename T> extern int v;    // expected-error {{redeclaration of 'v' with a different type: 'int' vs 'T'}}

#ifndef PRECXX11
    template<typename T> extern auto v;   // expected-error {{declaration of variable 'v' with deduced type 'auto' requires an initializer}}
#endif

    template<typename T> T var = T();     // expected-note {{previous definition is here}}
    extern int var;                       // expected-error {{redefinition of 'var' as different kind of symbol}}
  }

#ifndef PRECXX11
  namespace pvt_auto {
    template<typename T> auto v0; // expected-error {{declaration of variable 'v0' with deduced type 'auto' requires an initializer}}
    template<typename T> auto v1 = T();  // expected-note {{previous definition is here}}
    template<typename T> int v1;   // expected-error {{redefinition of 'v1' with a different type: 'int' vs 'auto'}}
    template<typename T> auto v2 = T();  // expected-note {{previous definition is here}}
    template<typename T> T v2;   // expected-error {{redefinition of 'v2'}}
    template<typename T> auto v3 = T();   // expected-note {{previous definition is here}}
    template<typename T> extern T v3;     // expected-error {{redeclaration of 'v3' with a different type: 'T' vs 'auto'}}
    template<typename T> auto v4 = T();
    template<typename T> extern auto v4;   // expected-error {{declaration of variable 'v4' with deduced type 'auto' requires an initializer}}
  }
#endif
  
}

namespace explicit_instantiation {
  template<typename T> 
  T pi0a = T(3.1415926535897932385);  // expected-note {{variable template 'pi0a' declared here}}
  template float pi0a<int>; // expected-error {{type 'float' of explicit instantiation of 'pi0a' does not match expected type 'int'}}

  template<typename T> 
  T pi0b = T(3.1415926535897932385);  // expected-note {{variable template 'pi0b' declared here}}
  template CONST int pi0b<int>; // expected-error {{type 'const int' of explicit instantiation of 'pi0b' does not match expected type 'int'}}

  template<typename T> 
  T pi0c = T(3.1415926535897932385);  // expected-note {{variable template 'pi0c' declared here}}
  template int pi0c<const int>;  // expected-error {{type 'int' of explicit instantiation of 'pi0c' does not match expected type 'const int'}}

  template<typename T> 
  T pi0 = T(3.1415926535897932385);
  template int pi0<int>;   // expected-note {{previous explicit instantiation is here}}
  template int pi0<int>;   // expected-error {{duplicate explicit instantiation of 'pi0<int>'}}

  template<typename T> 
  CONST T pi1a = T(3.1415926535897932385);  // expected-note {{variable template 'pi1a' declared here}}
  template int pi1a<int>; // expected-error {{type 'int' of explicit instantiation of 'pi1a' does not match expected type 'const int'}}

  template<typename T> 
  CONST T pi1b = T(3.1415926535897932385);  // expected-note {{variable template 'pi1b' declared here}}
  template int pi1b<const int>;  // expected-error {{type 'int' of explicit instantiation of 'pi1b' does not match expected type 'const const int'}}

  template<typename T> 
  CONST T pi1 = T(3.1415926535897932385);
  template CONST int pi1<int>;   // expected-note {{previous explicit instantiation is here}}
  template CONST int pi1<int>;   // expected-error {{duplicate explicit instantiation of 'pi1<int>'}}

#ifndef PRECXX11
  namespace auto_var {
    template<typename T> auto var0 = T();
    template auto var0<int>;    // expected-error {{'auto' variable template instantiation is not allowed}}
    
    template<typename T> auto var = T();
    template int var<int>;
  }
#endif

  template<typename=int> int missing_args; // expected-note {{here}}
  template int missing_args; // expected-error {{must specify a template argument list}}

  namespace extern_var {
    // TODO:
  }
}

namespace explicit_specialization {

  namespace good {
    template<typename T1, typename T2>
    CONST int pi2 = 1;

    template<typename T>
    CONST int pi2<T,int> = 2;

    template<typename T>
    CONST int pi2<int,T> = 3;

    template<> CONST int pi2<int,int> = 4;

#ifndef PRECXX11   
    void foo() {
      static_assert(pi2<int,int> == 4, "");
      static_assert(pi2<float,int> == 2, "");
      static_assert(pi2<int,float> == 3, "");
      static_assert(pi2<int,float> == pi2<int,double>, "");
      static_assert(pi2<float,float> == 1, "");
      static_assert(pi2<float,float> == pi2<float,double>, "");
    }
#endif
  }

  namespace ambiguous {

    template<typename T1, typename T2>
    CONST int pi2 = 1;

    template<typename T>
    CONST int pi2<T,int> = 2; // expected-note {{partial specialization matches [with T = int]}}

    template<typename T>
    CONST int pi2<int,T> = 3; // expected-note {{partial specialization matches [with T = int]}}
    
    void foo() {
      int a = pi2<int,int>;  // expected-error {{ambiguous partial specializations of 'pi2<int, int>'}}
    }
  }
  
  namespace type_changes {

    template<typename T> 
    T pi0 = T(3.1415926535897932385);

    template<> float pi0<int> = 10;
    template<> int pi0<const int> = 10;

    template<typename T>
    T pi1 = T(3.1415926535897932385);
    template<> CONST int pi1<int> = 10;

    template<typename T>
    T pi2 = T(3.1415926535897932385);
    template<> int pi2<const int> = 10;

    template<typename T>
    CONST T pi4 = T(3.1415926535897932385);
    template<> int pi4<int> = 10;
  }

  namespace redefinition {
    template<typename T>
    T pi0 = T(3.1415926535897932385);

    template<> int pi0<int> = 10;   // expected-note 3{{previous definition is here}}
#ifndef PRECXX11
// expected-note@-2 {{previous definition is here}}
#endif
    template<> int pi0<int> = 10;   // expected-error {{redefinition of 'pi0<int>'}}
    template<> CONST int pi0<int> = 10; // expected-error {{redefinition of 'pi0' with a different type: 'const int' vs 'int'}}
    template<> float pi0<int> = 10; // expected-error {{redefinition of 'pi0' with a different type: 'float' vs 'int'}}
#ifndef PRECXX11
    template<> auto pi0<int> = 10;  // expected-error {{redefinition of 'pi0<int>'}}
#endif


    template<typename T> 
    CONST T pi1 = T(3.1415926535897932385);

    template<> CONST int pi1<int> = 10;   // expected-note {{previous definition is here}}
    template<> CONST int pi1<int> = 10;   // expected-error {{redefinition of 'pi1<int>'}}
  }
  
  namespace before_instantiation {
    template<typename T> 
    T pi0 = T(3.1415926535897932385);   // expected-note {{variable template 'pi0' declared here}}

    template<> int pi0<int> = 10; // expected-note 2{{previous template specialization is here}}
    template int pi0<int>;        // expected-warning {{has no effect}}
    template float pi0<int>;      // expected-error {{type 'float' of explicit instantiation of 'pi0' does not match expected type}} expected-warning {{has no effect}}

    template<typename T1, typename T2>
    CONST int pi2 = 1;

    template<typename T> CONST int pi2<T,int> = 2;
    template CONST int pi2<int,int>;
  }
  namespace after_instantiation {
    template<typename T> 
    T pi0 = T(3.1415926535897932385);

    template int pi0<int>;   // expected-note 2{{explicit instantiation first required here}}
    template<> int pi0<int> = 10; // expected-error {{explicit specialization of 'pi0' after instantiation}}
    template<> float pi0<int>;    // expected-error {{explicit specialization of 'pi0' after instantiation}}

    template<typename T1, typename T2>
    CONST int pi2 = 1;

    template CONST int pi2<int,int>;
    template<typename T> CONST int pi2<T,int> = 2;
  }

#ifndef PRECXX11
  namespace auto_var {
    template<typename T, typename> auto var0 = T();
    template<typename T> auto var0<T,int> = T();
    template<> auto var0<int,int> = 7;

    template<typename T, typename> auto var = T();
    template<typename T> T var<T,int> = T(5);
    template<> int var<int,int> = 7;

    void foo() {
      int i0 = var0<int,int>;
      int b = var<int,int>;
    }
  }
#endif
  
  namespace extern_var {
    // TODO:
  }
  
  namespace diff_type {
    // TODO:
    template<typename T> T* var = new T();
#ifndef PRECXX11
    template<typename T> auto var<T*> = T();  // expected-note {{previous definition is here}}
    template<typename T> T var<T*> = T();     // expected-error {{redefinition of 'var' with a different type: 'T' vs 'auto'}}
#endif
  }
}

namespace narrowing {
  template<typename T> T v = {1234};  // expected-warning {{implicit conversion from 'int' to 'char' changes value from 1234 to}}
#ifndef PRECXX11
  // expected-error@-2 {{constant expression evaluates to 1234 which cannot be narrowed to type 'char'}}\
  // expected-note@-2 {{insert an explicit cast to silence this issue}}
#endif
  int k = v<char>;        // expected-note {{in instantiation of variable template specialization 'narrowing::v<char>' requested here}}
}

namespace use_in_structs {
  // TODO:
}

namespace attributes {
  // TODO:
}

#ifndef PRECXX11
namespace arrays {
  template<typename T>
  T* arr = new T[10]{T(10), T(23)};

  float f = 10.5;
  template<> float* arr<float> = &f;
  
  void bar() {
    int *iarr = arr<int>;
    iarr[0] = 1;
    iarr[2] = 3;
    iarr[6] = -2;

    float ff = *arr<float>;
    float nof = arr<float>[3];  // No bounds-check in C++
  }
}
#endif

namespace nested {
  
  namespace n0a {
    template<typename T> 
    T pi0a = T(3.1415926535897932385);
  }
  
  using namespace n0a;
  int i0a = pi0a<int>;
  
  template float pi0a<float>;
  float f0a = pi0a<float>;
  
  template<> double pi0a<double> = 5.2;
  double d0a = pi0a<double>;

  namespace n0b {
    template<typename T> 
    T pi0b = T(3.1415926535897932385);
  }
  
  int i0b = n0b::pi0b<int>;
  
  template float n0b::pi0b<float>;
  float f0b = n0b::pi0b<float>;
  
  template<> double n0b::pi0b<double> = 5.2;
  double d0b = n0b::pi0b<double>;
  
  namespace n1 {
    template<typename T> 
    T pi1a = T(3.1415926535897932385); // expected-note {{explicitly specialized declaration is here}}
#ifndef PRECXX11
// expected-note@-2 {{explicit instantiation refers here}}
#endif

    template<typename T> 
    T pi1b = T(3.1415926535897932385); // expected-note {{explicitly specialized declaration is here}}
#ifndef PRECXX11
// expected-note@-2 {{explicit instantiation refers here}}
#endif
  }
  
  namespace use_n1a {
    using namespace n1;
    int i1 = pi1a<int>;

    template float pi1a<float>;
#ifndef PRECXX11
// expected-error@-2 {{explicit instantiation of 'pi1a<float>' not in a namespace enclosing 'n1'}}
#endif
    float f1 = pi1a<float>;
    
    template<> double pi1a<double> = 5.2; // expected-error {{not in a namespace enclosing 'n1'}}
    double d1 = pi1a<double>;
  }
  
  namespace use_n1b {
    int i1 = n1::pi1b<int>;
    
    template float n1::pi1b<float>;
#ifndef PRECXX11
// expected-error@-2 {{explicit instantiation of 'pi1b<float>' not in a namespace enclosing 'n1'}}
#endif
    float f1 = n1::pi1b<float>;
    
    template<> double n1::pi1b<double> = 5.2;  // expected-error {{not in a namespace enclosing 'n1'}}
    double d1 = n1::pi1b<double>;
  }
}

namespace nested_name {
  template<typename T> int a; // expected-note {{variable template 'a' declared here}}
  a<int>::b c; // expected-error {{qualified name refers into a specialization of variable template 'a'}}

  class a<int> {}; // expected-error {{identifier followed by '<' indicates a class template specialization but 'a' refers to a variable template}}
  enum a<int> {}; // expected-error {{expected identifier or '{'}} expected-warning {{does not declare anything}}
}

namespace PR18530 {
  template<typename T> int a;
  int a<int>; // expected-error {{requires 'template<>'}}
}

namespace PR19152 {
#ifndef PRECXX11
  template<typename T> const auto x = 1;
  static_assert(x<int> == 1, "");
#endif
}

namespace PR19169 {
  template <typename T> int* f();
  template <typename T> void f();
  template<> int f<double>; // expected-error {{no variable template matches specialization; did you mean to use 'f' as function template instead?}}
  
  template <typename T> void g();
  template<> int g<double>; // expected-error {{no variable template matches specialization; did you mean to use 'g' as function template instead?}}
}

#ifndef PRECXX11
template <typename... Args> struct Variadic_t { };
template <typename... Args> Variadic_t<Args...> Variadic;
auto variadic1 = Variadic<>;
auto variadic2 = Variadic<int, int>;
#endif

namespace VexingParse {
  template <typename> int var; // expected-note {{declared here}}
  int x(var); // expected-error {{use of variable template 'var' requires template arguments}}
}
