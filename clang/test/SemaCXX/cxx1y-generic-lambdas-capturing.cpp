// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++2a -verify -fsyntax-only -fblocks -emit-llvm-only %s
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s -DDELAYED_TEMPLATE_PARSING
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fms-extensions %s -DMS_EXTENSIONS
// DONTRUNYET: %clang_cc1 -std=c++1y -verify -fsyntax-only -fblocks -fdelayed-template-parsing -fms-extensions %s -DMS_EXTENSIONS -DDELAYED_TEMPLATE_PARSING

constexpr int ODRUSE_SZ = sizeof(char);

template<class T, int N>
void f(T, const int (&)[N]) { }

template<class T>
void f(const T&, const int (&)[ODRUSE_SZ]) { }

#define DEFINE_SELECTOR(x)   \
  int selector_ ## x[sizeof(x) == ODRUSE_SZ ? ODRUSE_SZ : ODRUSE_SZ + 5]

#define F_CALL(x, a) f(x, selector_ ## a)

// This is a risky assumption, because if an empty class gets captured by value
// the lambda's size will still be '1' 
#define ASSERT_NO_CAPTURES(L) static_assert(sizeof(L) == 1, "size of closure with no captures must be 1")
#define ASSERT_CLOSURE_SIZE_EXACT(L, N) static_assert(sizeof(L) == (N), "size of closure must be " #N)
#define ASSERT_CLOSURE_SIZE(L, N) static_assert(sizeof(L) >= (N), "size of closure must be >=" #N)


namespace sample {
  struct X {  
    int i;
    X(int i) : i(i) { }
  };
} 
 
namespace test_transformations_in_templates {
template<class T> void foo(T t) {
  auto L = [](auto a) { return a; };
}
template<class T> void foo2(T t) {
  auto L = [](auto a) -> void { 
    auto M = [](char b) -> void {
      auto N = [](auto c) -> void { 
        int selector[sizeof(c) == 1 ? 
                      (sizeof(b) == 1 ? 1 : 2) 
                      : 2
                    ]{};      
      };  
      N('a');
    };    
  };
  L(3.14);
}

void doit() {
  foo(3);
  foo('a');
  foo2('A');
}
}

namespace test_return_type_deduction {

void doit() {

  auto L = [](auto a, auto b) {
    if ( a > b ) return a;
    return b;
  };
  L(2, 4);
  {
    auto L2 = [](auto a, int i) {
      return a + i;
    };
    L2(3.14, 2);
  }
  {
    int a; //expected-note{{declared here}}
    auto B = []() { return ^{ return a; }; }; //expected-error{{cannot be implicitly capture}}\
                                              //expected-note{{begins here}}
  //[](){ return ({int b = 5; return 'c'; 'x';}); };

  //auto X = ^{ return a; };
  
  //auto Y = []() -> auto { return 3; return 'c'; };

  }  
}  
}


namespace test_no_capture{
void doit() {
  const int x = 10; //expected-note{{declared here}}
  {
    // should not capture 'x' - variable undergoes lvalue-to-rvalue
    auto L = [=](auto a) {
      int y = x;
      return a + y;
    };
    ASSERT_NO_CAPTURES(L);
  }
  {
    // should not capture 'x' - even though certain instantiations require
    auto L = [](auto a) { //expected-note{{begins here}}
      DEFINE_SELECTOR(a);
      F_CALL(x, a); //expected-error{{'x' cannot be implicitly captured}}
    };
    ASSERT_NO_CAPTURES(L);
    L('s'); //expected-note{{in instantiation of}}
  }
  {
    // Does not capture because no default capture in inner most lambda 'b'
    auto L = [=](auto a) {
      return [=](int p) {
        return [](auto b) {
          DEFINE_SELECTOR(a);
          F_CALL(x, a); 
          return 0;        
        }; 
      };
    };
    ASSERT_NO_CAPTURES(L);
  }  
}  // doit
} // namespace

namespace test_capture_of_potentially_evaluated_expression {
void doit() {
  const int x = 5;
  {
    auto L = [=](auto a) {
      DEFINE_SELECTOR(a);
      F_CALL(x, a);
    };
    static_assert(sizeof(L) == 4, "Must be captured");
  }
  {
    int j = 0; //expected-note{{declared}}
    auto L = [](auto a) {  //expected-note{{begins here}}
      return j + 1; //expected-error{{cannot be implicitly captured}}
    };
  }
  {
    const int x = 10;
    auto L = [](auto a) {
      //const int y = 20;
      return [](int p) { 
        return [](auto b) { 
          DEFINE_SELECTOR(a);
          F_CALL(x, a);  
          return 0;        
        }; 
      };
    };
    auto M = L(3);
    auto N = M(5);
    
  }
  
  { // if the nested capture does not implicitly or explicitly allow any captures
    // nothing should capture - and instantiations will create errors if needed.
    const int x = 0;
    auto L = [=](auto a) { // <-- #A
      const int y = 0;
      return [](auto b) { // <-- #B
        int c[sizeof(b)];
        f(x, c);
        f(y, c);
        int i = x;
      };
    };
    ASSERT_NO_CAPTURES(L);
    auto M_int = L(2);
    ASSERT_NO_CAPTURES(M_int);
  }
  { // Permutations of this example must be thoroughly tested!
    const int x = 0;
    sample::X cx{5};
    auto L = [=](auto a) { 
      const int z = 3;
      return [&,a](auto b) {
        // expected-warning@-1 {{address of stack memory associated with local variable 'z' returned}}
        // expected-note@#call {{in instantiation of}}
        const int y = 5;
        return [=](auto c) {
          int d[sizeof(a) == sizeof(c) || sizeof(c) == sizeof(b) ? 2 : 1];
          f(x, d);
          f(y, d);
          f(z, d); // expected-note {{implicitly captured by reference due to use here}}
          decltype(a) A = a;
          decltype(b) B = b;
          const int &i = cx.i;
        }; 
      };
    };
    auto M = L(3)(3.5); // #call
    M(3.14);
  }
}
namespace Test_no_capture_of_clearly_no_odr_use {
auto foo() {
 const int x = 10; 
 auto L = [=](auto a) {
    return  [=](auto b) {
      return [=](auto c) {
        int A = x;
        return A;
      };
    };
  };
  auto M = L(1);
  auto N = M(2.14);
  ASSERT_NO_CAPTURES(L);
  ASSERT_NO_CAPTURES(N);
  
  return 0;
}
}

namespace Test_capture_of_odr_use_var {
auto foo() {
 const int x = 10; 
 auto L = [=](auto a) {
    return  [=](auto b) {
      return [=](auto c) {
        int A = x;
        const int &i = x;
        decltype(a) A2 = a;
        return A;
      };
    };
  };
  auto M_int = L(1);
  auto N_int_int = M_int(2);
  ASSERT_CLOSURE_SIZE_EXACT(L, sizeof(x));
  // M_int captures both a & x   
  ASSERT_CLOSURE_SIZE_EXACT(M_int, sizeof(x) + sizeof(int));
  // N_int_int captures both a & x   
  ASSERT_CLOSURE_SIZE_EXACT(N_int_int, sizeof(x) + sizeof(int)); 
  auto M_double = L(3.14);
  ASSERT_CLOSURE_SIZE(M_double, sizeof(x) + sizeof(double));
  
  return 0;
}
auto run = foo();
}

}    
namespace more_nested_captures_1 {
template<class T> struct Y {
  static void f(int, double, ...) { }
  template<class R> 
  static void f(const int&, R, ...) { }
  template<class R>
  void foo(R t) {
    const int x = 10; //expected-note{{declared here}}
    auto L = [](auto a) { 
       return [=](auto b) {
        return [=](auto c) { 
          f(x, c, b, a);  //expected-error{{reference to local variable 'x'}}
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3.14);
    N(5);  //expected-note{{in instantiation of}}
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); //expected-note{{in instantiation of}}
}


namespace more_nested_captures_1_1 {
template<class T> struct Y {
  static void f(int, double, ...) { }
  template<class R> 
  static void f(const int&, R, ...) { }
  template<class R>
  void foo(R t) {
    const int x = 10; //expected-note{{declared here}}
    auto L = [](auto a) { 
       return [=](char b) {
        return [=](auto c) { 
          f(x, c, b, a);  //expected-error{{reference to local variable 'x'}}
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3.14);
    N(5);  //expected-note{{in instantiation of}}
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); //expected-note{{in instantiation of}}
}
namespace more_nested_captures_1_2 {
template<class T> struct Y {
  static void f(int, double, ...) { }
  template<class R> 
  static void f(const int&, R, ...) { }
  template<class R>
  void foo(R t) {
    const int x = 10; 
    auto L = [=](auto a) { 
       return [=](char b) {
        return [=](auto c) { 
          f(x, c, b, a);  
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3.14);
    N(5);  
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); 
}

namespace more_nested_captures_1_3 {
template<class T> struct Y {
  static void f(int, double, ...) { }
  template<class R> 
  static void f(const int&, R, ...) { }
  template<class R>
  void foo(R t) {
    const int x = 10; //expected-note{{declared here}}
    auto L = [=](auto a) { 
       return [](auto b) {
        const int y = 0;
        return [=](auto c) { 
          f(x, c, b);  //expected-error{{reference to local variable 'x'}}
          f(y, b, c);
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3.14);
    N(5);  //expected-note{{in instantiation of}}
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); //expected-note{{in instantiation of}}
}


namespace more_nested_captures_1_4 {
template<class T> struct Y {
  static void f(int, double, ...) { }
  template<class R> 
  static void f(const int&, R, ...) { }
  template<class R>
  void foo(R t) {
    const int x = 10; //expected-note{{declared here}}
    auto L = [=](auto a) {
       T t2{t};       
       return [](auto b) {
        const int y = 0; //expected-note{{declared here}}
        return [](auto c) { //expected-note 2{{lambda expression begins here}}
          f(x, c);  //expected-error{{variable 'x'}}
          f(y, c);  //expected-error{{variable 'y'}}
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N_char = M('b');
    N_char(3.14);
    auto N_double = M(3.14);
    N_double(3.14);
    N_char(3);  //expected-note{{in instantiation of}}
  }
};
Y<int> yi;
int run = (yi.foo('a'), 0); //expected-note{{in instantiation of}}
}


namespace more_nested_captures_2 {
template<class T> struct Y {
  static void f(int, double) { }
  template<class R> 
  static void f(const int&, R) { }
  template<class R> 
  void foo(R t) {
    const int x = 10;
    auto L = [=](auto a) { 
       return [=](auto b) {
        return [=](auto c) { 
          f(x, c);  
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3);
    N(3.14);
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0);

}

namespace more_nested_captures_3 {
template<class T> struct Y {
  static void f(int, double) { }
  template<class R> 
  static void f(const int&, R) { }
  template<class R> 
  void foo(R t) {
    const int x = 10; //expected-note{{declared here}}
    auto L = [](auto a) { 
       return [=](auto b) {
        return [=](auto c) { 
          f(x, c);   //expected-error{{reference to local variable 'x'}}
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3); //expected-note{{in instantiation of}}
    N(3.14);
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); //expected-note{{in instantiation of}}

}

namespace more_nested_captures_4 {
template<class T> struct Y {
  static void f(int, double) { }
  template<class R> 
  static void f(const int&, R) { }
  template<class R> 
  void foo(R t) {
    const int x = 10;  //expected-note{{'x' declared here}}
    auto L = [](auto a) { 
       return [=](char b) {
        return [=](auto c) { 
          f(x, c);  //expected-error{{reference to local variable 'x'}}
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3); //expected-note{{in instantiation of}}
    N(3.14);
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0); //expected-note{{in instantiation of}}

}

namespace more_nested_captures_5 {
template<class T> struct Y {
  static void f(int, double) { }
  template<class R> 
  static void f(const int&, R) { }
  template<class R> 
  void foo(R t) {
    const int x = 10;
    auto L = [=](auto a) { 
       return [=](char b) {
        return [=](auto c) { 
          f(x, c);   
          return 0; 
        };
      };
    };
    auto M = L(t);
    auto N = M('b');
    N(3); 
    N(3.14);
  }
};
Y<int> yi;
int run = (yi.foo(3.14), 0);

}

namespace lambdas_in_NSDMIs {
template<class T>
  struct L {
      T t{};
      T t2 = ([](auto a) { return [](auto b) { return b; };})(t)(t);    
      T t3 = ([](auto a) { return a; })(t);    
  };
  L<int> l; 
  int run = l.t2; 
}
namespace test_nested_decltypes_in_trailing_return_types {
int foo() {
  auto L = [](auto a) {
      return [](auto b, decltype(a) b2) -> decltype(a) {
        return decltype(a){};
      };
  };
  auto M = L(3.14);
  M('a', 6.26);
  return 0;
}
}

namespace more_this_capture_1 {
struct X {
  void f(int) { }
  static void f(double) { }
  void foo() {
    {
      auto L = [=](auto a) {
        f(a);
      };
      L(3);
      L(3.13);
    }
    {
      auto L = [](auto a) {
        f(a); //expected-error{{this}}
      };
      L(3.13);
      L(2); //expected-note{{in instantiation}}
    }
  }
  
  int g() {
    auto L = [=](auto a) { 
      return [](int i) {
        return [=](auto b) {
          f(b); 
          int x = i;
        };
      };
    };
    auto M = L(0.0); 
    auto N = M(3);
    N(5.32); // OK 
    return 0;
  }
};
int run = X{}.g();
}
namespace more_this_capture_1_1 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [](int i) {
        return [=](auto b) {
          f(decltype(a){}); //expected-error{{this}}
          int x = i;
        };
      };
    };
    auto M = L(0.0);  
    auto N = M(3);
    N(5.32); // OK 
    L(3); // expected-note{{instantiation}}
    return 0;
  }
};
int run = X{}.g();
}

namespace more_this_capture_1_1_1 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [](auto b) {
        return [=](int i) {
          f(b); 
          f(decltype(a){}); //expected-error{{this}}
        };
      };
    };
    auto M = L(0.0);  // OK
    auto N = M(3.3); //OK
    auto M_int = L(0); //expected-note{{instantiation}}
    return 0;
  }
};
int run = X{}.g();
}


namespace more_this_capture_1_1_1_1 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [](auto b) {
        return [=](int i) {
          f(b); //expected-error{{this}}
          f(decltype(a){}); 
        };
      };
    };
    auto M_double = L(0.0);  // OK
    auto N = M_double(3); //expected-note{{instantiation}}
    
    return 0;
  }
};
int run = X{}.g();
}

namespace more_this_capture_2 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [](int i) {
        return [=](auto b) {
          f(b); //expected-error{{'this' cannot}}
          int x = i;
        };
      };
    };
    auto M = L(0.0); 
    auto N = M(3);
    N(5); // NOT OK expected-note{{in instantiation of}}
    return 0;
  }
};
int run = X{}.g();
}
namespace diagnose_errors_early_in_generic_lambdas {

int foo()
{

  { // This variable is used and must be caught early, do not need instantiation
    const int x = 0; //expected-note{{declared}}
    auto L = [](auto a) { //expected-note{{begins}}
      const int &r = x; //expected-error{{variable}}      
    };
  }
  { // This variable is not used 
    const int x = 0; 
    auto L = [](auto a) { 
      int i = x;       
    };
  }
  { 
  
    const int x = 0; //expected-note{{declared}}
    auto L = [=](auto a) { // <-- #A
      const int y = 0;
      return [](auto b) { //expected-note{{begins}}
        int c[sizeof(b)];
        f(x, c);
        f(y, c);
        int i = x;
        // This use will always be an error regardless of instantatiation
        // so diagnose this early.
        const int &r = x; //expected-error{{variable}}
      };
    };
    
  }
  return 0;
}

int run = foo();
}

namespace generic_nongenerics_interleaved_1 {
int foo() {
  {
    auto L = [](int a) {
      int y = 10;
      return [=](auto b) { 
        return a + y;
      };
    };
    auto M = L(3);
    M(5);
  }
  {
    int x;
    auto L = [](int a) {
      int y = 10;
      return [=](auto b) { 
        return a + y;
      };
    };
    auto M = L(3);
    M(5);
  }
  {
    // FIXME: why are there 2 error messages here?
    int x;
    auto L = [](auto a) { //expected-note {{declared here}}
      int y = 10; //expected-note {{declared here}}
      return [](int b) { //expected-note 2{{expression begins here}}
        return [=] (auto c) {
          return a + y; //expected-error 2{{cannot be implicitly captured}}
        };
      };
    };
  }
  {
    int x;
    auto L = [](auto a) { 
      int y = 10; 
      return [=](int b) { 
        return [=] (auto c) {
          return a + y; 
        };
      };
    };
  }
  return 1;
}

int run = foo();
}
namespace dont_capture_refs_if_initialized_with_constant_expressions {

auto foo(int i) {
  // This is surprisingly not odr-used within the lambda!
  static int j;
  j = i;
  int &ref_j = j;
  return [](auto a) { return ref_j; }; // ok
}

template<class T>
auto foo2(T t) {
  // This is surprisingly not odr-used within the lambda!
  static T j;
  j = t;
  T &ref_j = j;
  return [](auto a) { return ref_j; }; // ok
}

int do_test() {
  auto L = foo(3);
  auto L_int = L(3);
  auto L_char = L('a');
  auto L1 = foo2(3.14);
  auto L1_int = L1(3);
  auto L1_char = L1('a');
  return 0;
}

} // dont_capture_refs_if_initialized_with_constant_expressions

namespace test_conversion_to_fptr {

template<class T> struct X {

  T (*fp)(T) = [](auto a) { return a; };
  
};

X<int> xi;

template<class T> 
void fooT(T t, T (*fp)(T) = [](auto a) { return a; }) {
  fp(t);
}

int test() {
{
  auto L = [](auto a) { return a; };
  int (*fp)(int) = L;
  fp(5);
  L(3);
  char (*fc)(char) = L;
  fc('b');
  L('c');
  double (*fd)(double) = L;
  fd(3.14);
  fd(6.26);
  L(4.25);
}
{
  auto L = [](auto a) ->int { return a; }; //expected-note 2{{candidate template ignored}}
  int (*fp)(int) = L;
  char (*fc)(char) = L; //expected-error{{no viable conversion}}
  double (*fd)(double) = L; //expected-error{{no viable conversion}}
}
{
  int x = 5;
  auto L = [=](auto b, char c = 'x') {
    int i = x;
    return [](auto a) ->decltype(a) { return a; };
  };
  int (*fp)(int) = L(8);
  fp(5);
  L(3);
  char (*fc)(char) = L('a');
  fc('b');
  L('c');
  double (*fd)(double) = L(3.14);
  fd(3.14);
  fd(6.26);

}
{
 auto L = [=](auto b) {
    return [](auto a) ->decltype(b)* { return (decltype(b)*)0; };
  };
  int* (*fp)(int) = L(8);
  fp(5);
  L(3);
  char* (*fc)(char) = L('a');
  fc('b');
  L('c');
  double* (*fd)(double) = L(3.14);
  fd(3.14);
  fd(6.26);
}
{
 auto L = [=](auto b) {
    return [](auto a) ->decltype(b)* { return (decltype(b)*)0; }; //expected-note{{candidate template ignored}}
  };
  char* (*fp)(int) = L('8');
  fp(5);
  char* (*fc)(char) = L('a');
  fc('b');
  double* (*fi)(int) = L(3.14);
  fi(5);
  int* (*fi2)(int) = L(3.14); //expected-error{{no viable conversion}}
}

{
 auto L = [=](auto b) {
    return [](auto a) { 
      return [=](auto c) { 
        return [](auto d) ->decltype(a + b + c + d) { return d; }; 
      }; 
    }; 
  };
  int (*fp)(int) = L('8')(3)(short{});
  double (*fs)(char) = L(3.14)(short{})('4');
}

  fooT(3);
  fooT('a');
  fooT(3.14);
  fooT("abcdefg");
  return 0;
}
int run2 = test();

}


namespace this_capture {
void f(char, int) { }
template<class T> 
void f(T, const int&) { }

struct X {
  int x = 0;
  void foo() {
    auto L = [=](auto a) {
         return [=](auto b) {
            //f(a, x++);
            x++;
         };
    };
    L('a')(5);
    L('b')(4);
    L(3.14)('3');
    
  }

};

int run = (X{}.foo(), 0);

namespace this_capture_unresolvable {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto lam = [=](auto a) { f(a); }; // captures 'this'
    lam(0); // ok.
    lam(0.0); // ok.
    return 0;
  }
  int g2() {
    auto lam = [](auto a) { f(a); }; // expected-error{{'this'}}
    lam(0); // expected-note{{in instantiation of}}
    lam(0.0); // ok.
    return 0;
  }
  double (*fd)(double) = [](auto a) { f(a); return a; };
  
};

int run = X{}.g();

}

namespace check_nsdmi_and_this_capture_of_member_functions {

struct FunctorDouble {
  template<class T> FunctorDouble(T t) { t(2.14); };
};
struct FunctorInt {
  template<class T> FunctorInt(T t) { t(2); }; //expected-note{{in instantiation of}}
};

template<class T> struct YUnresolvable {
  void f(int) { }
  static void f(double) { }
  
  T t = [](auto a) { f(a); return a; }; 
  T t2 = [=](auto b) { f(b); return b; };
};

template<class T> struct YUnresolvable2 {
  void f(int) { }
  static void f(double) { }
  
  T t = [](auto a) { f(a); return a; }; //expected-error{{'this'}} \
                                        //expected-note{{in instantiation of}}
  T t2 = [=](auto b) { f(b); return b; };
};


YUnresolvable<FunctorDouble> yud;
// This will cause an error since it call's with an int and calls a member function.
YUnresolvable2<FunctorInt> yui;


template<class T> struct YOnlyStatic {
  static void f(double) { }
  
  T t = [](auto a) { f(a); return a; };
};
YOnlyStatic<FunctorDouble> yos;
template<class T> struct YOnlyNonStatic {
  void f(int) { }
  
  T t = [](auto a) { f(a); return a; }; //expected-error{{'this'}}
};


}


namespace check_nsdmi_and_this_capture_of_data_members {

struct FunctorDouble {
  template<class T> FunctorDouble(T t) { t(2.14); };
};
struct FunctorInt {
  template<class T> FunctorInt(T t) { t(2); }; 
};

template<class T> struct YThisCapture {
  const int x = 10;
  static double d; 
  T t = [](auto a) { return x; }; //expected-error{{'this'}}
  T t2 = [](auto b) {  return d; };
  T t3 = [this](auto a) {
          return [=](auto b) {
            return x;
         };
  };
  T t4 = [=](auto a) {
          return [=](auto b) {
            return x;
         };
  };
  T t5 = [](auto a) {
          return [=](auto b) {
            return x;  //expected-error{{'this'}}
         };
  };
};

template<class T> double YThisCapture<T>::d = 3.14;


}


#ifdef DELAYED_TEMPLATE_PARSING
template<class T> void foo_no_error(T t) { 
  auto L = []()  
    { return t; }; 
}
template<class T> void foo(T t) { //expected-note 2{{declared here}}
  auto L = []()  //expected-note 2{{begins here}}
    { return t; }; //expected-error 2{{cannot be implicitly captured}}
}
template void foo(int); //expected-note{{in instantiation of}}

#else

template<class T> void foo(T t) { //expected-note{{declared here}}
  auto L = []()  //expected-note{{begins here}}
    { return t; }; //expected-error{{cannot be implicitly captured}}
}

#endif
}

namespace no_this_capture_for_static {

struct X {
  static void f(double) { }
  
  int g() {
    auto lam = [=](auto a) { f(a); }; 
    lam(0); // ok.
    ASSERT_NO_CAPTURES(lam);
    return 0;
  }
};

int run = X{}.g();
}

namespace this_capture_for_non_static {

struct X {
  void f(double) { }
  
  int g() {
    auto L = [=](auto a) { f(a); }; 
    L(0); 
    auto L2 = [](auto a) { f(a); }; //expected-error {{cannot be implicitly captured}}
    return 0;
  }
};

int run = X{}.g();
}

namespace this_captures_with_num_args_disambiguation {

struct X {
  void f(int) { }
  static void f(double, int i) { }
  int g() {
    auto lam = [](auto a) { f(a, a); }; 
    lam(0);
    return 0;
  }
};

int run = X{}.g();
}
namespace enclosing_function_is_template_this_capture {
// Only error if the instantiation tries to use the member function.
struct X {
  void f(int) { }
  static void f(double) { }
  template<class T>
  int g(T t) {
    auto L = [](auto a) { f(a); }; //expected-error{{'this'}} 
    L(t); // expected-note{{in instantiation of}}
    return 0;
  }
};

int run = X{}.g(0.0); // OK.
int run2 = X{}.g(0);  // expected-note{{in instantiation of}}


}

namespace enclosing_function_is_template_this_capture_2 {
// This should error, even if not instantiated, since
// this would need to be captured.
struct X {
  void f(int) { }
  template<class T>
  int g(T t) {
    auto L = [](auto a) { f(a); }; //expected-error{{'this'}} 
    L(t); 
    return 0;
  }
};

}


namespace enclosing_function_is_template_this_capture_3 {
// This should not error, this does not need to be captured.
struct X {
  static void f(int) { }
  template<class T>
  int g(T t) {
    auto L = [](auto a) { f(a); };  
    L(t); 
    return 0;
  }
};

int run = X{}.g(0.0); // OK.
int run2 = X{}.g(0);  // OK.

}

namespace nested_this_capture_1 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [this]() {
        return [=](auto b) {
          f(b); 
        };
      };
    };
    auto M = L(0);
    auto N = M();
    N(5);    
    return 0;
  }
};

int run = X{}.g();

}


namespace nested_this_capture_2 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [&]() {
        return [=](auto b) {
          f(b);  
        };
      };
    };
    auto M = L(0);
    auto N = M();
    N(5);   
    N(3.14);    
    return 0;
  }
};

int run = X{}.g();

}

namespace nested_this_capture_3_1 {
struct X {
  template<class T>
  void f(int, T t) { }
  template<class T>
  static void f(double, T t) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [&](auto c) {
        return [=](auto b) {
          f(b, c); 
        };
      };
    };
    auto M = L(0);
    auto N = M('a');
    N(5); 
    N(3.14);    
    return 0;
  }
};

int run = X{}.g();

}


namespace nested_this_capture_3_2 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [=](auto a) { 
      return [](int i) {
        return [=](auto b) {
          f(b); //expected-error {{'this' cannot}}
          int x = i;
        };
      };
    };
    auto M = L(0.0); 
    auto N = M(3);
    N(5); //expected-note {{in instantiation of}}
    N(3.14); // OK.    
    return 0;
  }
};

int run = X{}.g();

}

namespace nested_this_capture_4 {
struct X {
  void f(int) { }
  static void f(double) { }
  
  int g() {
    auto L = [](auto a) { 
      return [=](auto i) {
        return [=](auto b) {
          f(b); //expected-error {{'this' cannot}}
          int x = i;
        };
      };
    };
    auto M = L(0.0); 
    auto N = M(3);
    N(5); //expected-note {{in instantiation of}}
    N(3.14); // OK.    
    return 0;
  }
};

int run = X{}.g();

}
namespace capture_enclosing_function_parameters {


inline auto foo(int x) {
  int i = 10;
  auto lambda = [=](auto z) { return x + z; };
  return lambda;
}

int foo2() {
  auto L = foo(3);
  L(4);
  L('a');
  L(3.14);
  return 0;
}

inline auto foo3(int x) {
  int local = 1;
  auto L = [=](auto a) {
        int i = a[local];    
        return  [=](auto b) mutable {
          auto n = b;
          return [&, n](auto c) mutable {
            ++local;
            return ++x;
          };
        };
  };
  auto M = L("foo-abc");
  auto N = M("foo-def");
  auto O = N("foo-ghi");
  
  return L;
}

int main() {
  auto L3 = foo3(3);
  auto M3 = L3("L3-1");
  auto N3 = M3("M3-1");
  auto O3 = N3("N3-1");
  N3("N3-2");
  M3("M3-2");
  M3("M3-3");
  L3("L3-2");
}
} // end ns

namespace capture_arrays {

inline int sum_array(int n) {
  int array2[5] = { 1, 2, 3, 4, 5};
  
  auto L = [=](auto N) -> int {  
    int sum = 0;
    int array[5] = { 1, 2, 3, 4, 5 };
    sum += array2[sum];
    sum += array2[N];    
    return 0;
  };
  L(2);
  return L(n);
}
}

namespace capture_non_odr_used_variable_because_named_in_instantiation_dependent_expressions {

// even though 'x' is not odr-used, it should be captured.

int test() {
  const int x = 10;
  auto L = [=](auto a) {
    (void) +x + a;
  };
  ASSERT_CLOSURE_SIZE_EXACT(L, sizeof(x));
}

} //end ns
#ifdef MS_EXTENSIONS
namespace explicit_spec {
template<class R> struct X {
  template<class T> int foo(T t) {
    auto L = [](auto a) { return a; };
    L(&t);
    return 0;
  }
  
  template<> int foo<char>(char c) { //expected-warning{{explicit specialization}}
    const int x = 10;
    auto LC = [](auto a) { return a; };
    R r;
    LC(&r);
    auto L = [=](auto a) {
      return [=](auto b) {
        int d[sizeof(a)];
        f(x, d);
      };
    };
    auto M = L(1);
    
    ASSERT_NO_CAPTURES(M);
    return 0;
  }
  
}; 

int run_char = X<int>{}.foo('a');
int run_int = X<double>{}.foo(4);
}
#endif // MS_EXTENSIONS

namespace nsdmi_capturing_this {
struct X {
  int m = 10;
  int n = [this](auto) { return m; }(20);
};

template<class T>
struct XT {
  T m = 10;
  T n = [this](auto) { return m; }(20);
};

XT<int> xt{};


}

void PR33318(int i) {
  [&](auto) { static_assert(&i != nullptr, ""); }(0); // expected-warning 2{{always true}} expected-note {{instantiation}}
}

// Check to make sure that we don't capture when member-calls are made to members that are not of 'this' class.
namespace PR34266 {
// https://bugs.llvm.org/show_bug.cgi?id=34266
namespace ns1 {
struct A {
  static void bar(int) { }
  static void bar(double) { }
};

struct B 
{
  template<class T>
  auto f() {
    auto L =  [=] { 
      T{}.bar(3.0); 
      T::bar(3);
    
    };
    ASSERT_NO_CAPTURES(L);
    return L;
  };
};

void test() {
  B{}.f<A>();
}
} // end ns1

namespace ns2 {
struct A {
  static void bar(int) { }
  static void bar(double) { }
};

struct B 
{
  using T = A;
  auto f() {
    auto L =  [=](auto a) {  
      T{}.bar(a); 
      T::bar(a);
    
    };
    ASSERT_NO_CAPTURES(L);
    return L;
  };
};

void test() {
  B{}.f()(3.0);
  B{}.f()(3);
}
} // end ns2

namespace ns3 {
struct A {
  void bar(int) { }
  static void bar(double) { }
};

struct B 
{
  using T = A;
  auto f() {
    auto L =  [=](auto a) { 
      T{}.bar(a); 
      T::bar(a); // This call ignores the instance member function because the implicit object argument fails to convert.
    
    };
    ASSERT_NO_CAPTURES(L);
    return L;
  };
};

void test() {
  B{}.f()(3.0);
  B{}.f()(3); 
}

} // end ns3


namespace ns4 {
struct A {
  void bar(int) { }
  static void bar(double) { }
};

struct B : A
{
  using T = A;
  auto f() {
    auto L =  [=](auto a) { 
      T{}.bar(a); 
      T::bar(a); 
    
    };
    // just check to see if the size if >= 2 bytes (which should be the case if we capture anything)
    ASSERT_CLOSURE_SIZE(L, 2);
    return L;
  };
};

void test() {
  B{}.f()(3.0);
  B{}.f()(3); 
}

} // end ns4

namespace ns5 {
struct A {
  void bar(int) { }
  static void bar(double) { }
};

struct B 
{
  template<class T>
  auto f() {
    auto L =  [&](auto a) { 
      T{}.bar(a); 
      T::bar(a); 
    
    };
    
    ASSERT_NO_CAPTURES(L);
    return L;
  };
};

void test() {
  B{}.f<A>()(3.0);
  B{}.f<A>()(3); 
}

} // end ns5

} // end PR34266

namespace capture_pack {
#if __cplusplus >= 201702L
  constexpr
#endif
  auto v =
    [](auto ...a) {
      [&](auto ...b) {
        ((a = b), ...); // expected-warning 0-1{{extension}}
      }(100, 20, 3);
      return (a + ...); // expected-warning 0-1{{extension}}
    }(400, 50, 6);
#if __cplusplus >= 201702L
  static_assert(v == 123);
#endif
}
