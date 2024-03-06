// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx23 -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 %s

class X { };

X operator+(X, X);

void f(X x) {
  x = x + x;
}

struct Y;
struct Z;

struct Y {
  Y(const Z&);
};

struct Z {
  Z(const Y&);
};

Y operator+(Y, Y);
bool operator-(Y, Y); // expected-note{{candidate function}}
bool operator-(Z, Z); // expected-note{{candidate function}}

void g(Y y, Z z) {
  y = y + z;
  bool b = y - z; // expected-error{{use of overloaded operator '-' is ambiguous}}
}

struct A {
  bool operator==(Z&); // expected-note 2{{candidate function}}
};

A make_A();

bool operator==(A&, Z&); // expected-note 3{{candidate function}} \
                         // cxx23-note 2{{candidate function}}


void h(A a, const A ac, Z z) {
  make_A() == z; // expected-warning{{equality comparison result unused}}
  a == z; // expected-error{{use of overloaded operator '==' is ambiguous}}
  ac == z; // expected-error{{invalid operands to binary expression ('const A' and 'Z')}}
}

struct B {
  bool operator==(const B&) const;

  void test(Z z) {
    make_A() == z; // expected-warning{{equality comparison result unused}}
  }
};

// we shouldn't see warnings about self-comparison,
// this is a member function, we dunno what it'll do
bool i(B b)
{
  return b == b;
}

enum Enum1 { };
enum Enum2 { };

struct E1 {
  E1(Enum1) { }
};

struct E2 {
  E2(Enum2);
};

// C++ [over.match.oper]p3 - enum restriction.
float& operator==(E1, E2);  // expected-note{{candidate function}} \
                            // cxx23-note{{candidate function}}


void enum_test(Enum1 enum1, Enum2 enum2, E1 e1, E2 e2, Enum1 next_enum1) {
  float &f1 = (e1 == e2);
  float &f2 = (enum1 == e2); 
  float &f3 = (e1 == enum2); 
  float &f4 = (enum1 == next_enum1);  // expected-error{{non-const lvalue reference to type 'float' cannot bind to a temporary of type 'bool'}}
}

// PR5244 - Argument-dependent lookup would include the two operators below,
// which would break later assumptions and lead to a crash.
class pr5244_foo
{
  pr5244_foo(int);
  pr5244_foo(char);
};

bool operator==(const pr5244_foo& s1, const pr5244_foo& s2); // expected-note{{candidate function}}
bool operator==(char c, const pr5244_foo& s); // expected-note{{candidate function}} \
                                              // cxx23-note{{candidate function}}

enum pr5244_bar
{
    pr5244_BAR
};

class pr5244_baz
{
public:
    pr5244_bar quux;
};

void pr5244_barbaz()
{
  pr5244_baz quuux;
  (void)(pr5244_BAR == quuux.quux);
}



struct PostInc {
  PostInc operator++(int);
  PostInc& operator++();
};

struct PostDec {
  PostDec operator--(int);
  PostDec& operator--();
};

void incdec_test(PostInc pi, PostDec pd) {
  const PostInc& pi1 = pi++;
  const PostDec& pd1 = pd--;
  PostInc &pi2 = ++pi;
  PostDec &pd2 = --pd;
}

struct SmartPtr {
  int& operator*();
  long& operator*() const volatile;
};

void test_smartptr(SmartPtr ptr, const SmartPtr cptr, 
                   const volatile SmartPtr cvptr) { // cxx23-warning {{volatile-qualified parameter type 'const volatile SmartPtr' is deprecated}}
  int &ir = *ptr;
  long &lr = *cptr;
  long &lr2 = *cvptr;
}


struct ArrayLike {
  int& operator[](int);
};

void test_arraylike(ArrayLike a) {
  int& ir = a[17];
}

struct SmartRef {
  int* operator&();
};

void test_smartref(SmartRef r) {
  int* ip = &r;
}

bool& operator,(X, Y);

void test_comma(X x, Y y) {
  bool& b1 = (x, y);
  X& xr = (x, x); // expected-warning {{left operand of comma operator has no effect}}
}

struct Callable {
  int& operator()(int, double = 2.71828); // expected-note{{candidate function}}
  float& operator()(int, double, long, ...); // expected-note{{candidate function}}

  double& operator()(float); // expected-note{{candidate function}}
};

struct Callable2 {
  int& operator()(int i = 0);
  double& operator()(...) const;
};

struct DerivesCallable : public Callable {
};

void test_callable(Callable c, Callable2 c2, const Callable2& c2c,
                   DerivesCallable dc) {
  int &ir = c(1);
  float &fr = c(1, 3.14159, 17, 42);

  c(); // expected-error{{no matching function for call to object of type 'Callable'}}

  double &dr = c(1.0f);

  int &ir2 = c2();
  int &ir3 = c2(1);
  double &fr2 = c2c();
  
  int &ir4 = dc(17);
  double &fr3 = dc(3.14159f);
}

typedef float FLOAT;
typedef int& INTREF;
typedef INTREF Func1(FLOAT, double);
typedef float& Func2(int, double);

struct ConvertToFunc {
  operator Func1*(); // expected-note 2{{conversion candidate of type 'INTREF (*)(FLOAT, double)'}}
  operator Func2&(); // expected-note 2{{conversion candidate of type 'float &(&)(int, double)'}}
  void operator()();
};

struct ConvertToFuncDerived : ConvertToFunc { };

void test_funcptr_call(ConvertToFunc ctf, ConvertToFuncDerived ctfd) {
  int &i1 = ctf(1.0f, 2.0);
  float &f1 = ctf((short int)1, 1.0f);
  ctf((long int)17, 2.0); // expected-error{{call to object of type 'ConvertToFunc' is ambiguous}}
  ctf();

  int &i2 = ctfd(1.0f, 2.0);
  float &f2 = ctfd((short int)1, 1.0f);
  ctfd((long int)17, 2.0); // expected-error{{call to object of type 'ConvertToFuncDerived' is ambiguous}}
  ctfd();
}

struct HasMember {
  int m;
};

struct Arrow1 {
  HasMember* operator->();
};

struct Arrow2 {
  Arrow1 operator->(); // expected-note{{candidate function}}
};

void test_arrow(Arrow1 a1, Arrow2 a2, const Arrow2 a3) {
  int &i1 = a1->m;
  int &i2 = a2->m;
  a3->m; // expected-error{{no viable overloaded 'operator->'}}
}

struct CopyConBase {
};

struct CopyCon : public CopyConBase {
  CopyCon(const CopyConBase &Base);

  CopyCon(const CopyConBase *Base) {
    *this = *Base;
  }
};

namespace N {
  struct X { };
}

namespace M {
  N::X operator+(N::X, N::X);
}

namespace M {
  void test_X(N::X x) {
    (void)(x + x);
  }
}

struct AA { bool operator!=(AA&); };
struct BB : AA {};
bool x(BB y, BB z) { return y != z; }


struct AX { 
  AX& operator ->();	 // expected-note {{declared here}}
  int b;
}; 

void m() {
  AX a; 
  a->b = 0; // expected-error {{circular pointer delegation detected}}
}

struct CircA {
  struct CircB& operator->(); // expected-note {{declared here}}
  int val;
};
struct CircB {
  struct CircC& operator->(); // expected-note {{declared here}}
};
struct CircC {
  struct CircA& operator->(); // expected-note {{declared here}}
};

void circ() {
  CircA a;
  a->val = 0; // expected-error {{circular pointer delegation detected}}
}

// PR5360: Arrays should lead to built-in candidates for subscript.
typedef enum {
  LastReg = 23,
} Register;
class RegAlloc {
  int getPriority(Register r) {
    return usepri[r];
  }
  int usepri[LastReg + 1];
};

// PR5546: Don't generate incorrect and ambiguous overloads for multi-level
// arrays.
namespace pr5546
{
  enum { X };
  extern const char *const sMoveCommands[][2][2];
  const char* a() { return sMoveCommands[X][0][0]; }
  const char* b() { return (*(sMoveCommands+X))[0][0]; }
}

// PR5512 and its discussion
namespace pr5512 {
  struct Y {
    operator short();
    operator float();
  };
  void g_test(Y y) {
    short s = 0;
    // DR507, this should be ambiguous, but we special-case assignment
    s = y;
    // Note: DR507, this is ambiguous as specified
    //s += y;
  }

  struct S {};
  void operator +=(int&, S);
  void f(S s) {
    int i = 0;
    i += s;
  }

  struct A {operator int();};
  int a;
  void b(A x) {
    a += x;
  }
}

// PR5900
namespace pr5900 {
  struct NotAnArray {};
  void test0() {
    NotAnArray x;
    x[0] = 0; // expected-error {{does not provide a subscript operator}}
  }

  struct NonConstArray {
    int operator[](unsigned); // expected-note {{candidate}}
  };
  int test1() {
    const NonConstArray x = NonConstArray();
    return x[0]; // expected-error {{no viable overloaded operator[] for type}}
  }

  // Not really part of this PR, but implemented at the same time.
  struct NotAFunction {};
  void test2() {
    NotAFunction x;
    x(); // expected-error {{does not provide a call operator}}
  }
}

// Operator lookup through using declarations.
namespace N {
  struct X2 { };
}

namespace N2 {
  namespace M {
    namespace Inner {
      template<typename T>
      N::X2 &operator<<(N::X2&, const T&);
    }
    using Inner::operator<<;
  }
}

void test_lookup_through_using() {
  using namespace N2::M;
  N::X2 x;
  x << 17;
}

namespace rdar9136502 {
  struct X {
    int i(); // expected-note{{possible target for call}}
    int i(int); // expected-note{{possible target for call}}
  };

  struct Y {
    Y &operator<<(int);
  };

  void f(X x, Y y) {
    y << x
      .i; // expected-error{{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  }
}

namespace rdar9222009 {
class StringRef {
  inline bool operator==(StringRef LHS, StringRef RHS) { // expected-error{{overloaded 'operator==' must be a binary operator (has 3 parameters)}}
    return !(LHS == RHS); // expected-error{{invalid operands to binary expression ('StringRef' and 'StringRef')}}
  }
};

}

namespace PR11784 {
  struct A { A& operator=(void (*x)()); };
  void f();
  void f(int);
  void g() { A x; x = f; }
}

namespace test10 {
  struct A {
    void operator[](float (*fn)(int)); // expected-note 2 {{not viable: no overload of 'bar' matching 'float (*)(int)'}}
  };

  float foo(int);
  float foo(float);

  template <class T> T bar(T);
  template <class T, class U> T bar(U);

  void test(A &a) {
    a[&foo];
    a[foo];

    a[&bar<int>]; // expected-error {{no viable overloaded operator[]}}
    a[bar<int>]; // expected-error {{no viable overloaded operator[]}}

    // If these fail, it's because we're not letting the overload
    // resolution for operator| resolve the overload of 'bar'.
    a[&bar<float>];
    a[bar<float>];
  }
}

struct InvalidOperatorEquals {
  InvalidOperatorEquals operator=() = delete; // expected-error {{overloaded 'operator=' must be a binary operator}}
};

namespace PR7681 {
  template <typename PT1, typename PT2> class PointerUnion;
  void foo(PointerUnion<int*, float*> &Result) {
    Result = 1; // expected-error {{no viable overloaded '='}} // expected-note {{type 'PointerUnion<int *, float *>' is incomplete}}
  }
}

namespace PR14995 {
  struct B {};
  template<typename ...T> void operator++(B, T...) {}

  void f() {
    B b;
    b++;  // ok
    ++b;  // ok
  }

  template<typename... T>
  struct C {
    void operator-- (T...) {}
  };

  void g() {
    C<int> postfix;
    C<> prefix;
    postfix--;  // ok
    --prefix;  // ok
  }

  struct D {};
  template<typename T> void operator++(D, T) {}

  void h() {
    D d;
    d++;  // ok
    ++d; // expected-error{{cannot increment value of type 'D'}}
  }

  template<typename...T> struct E {
    void operator++(T...) {} // expected-error{{parameter of overloaded post-increment operator must have type 'int' (not 'char')}}
  };

  E<char> e; // expected-note {{in instantiation of template class 'PR14995::E<char>' requested here}}
  
  struct F {
    template<typename... T>
    int operator++ (T...) {}
  };

  int k1 = F().operator++(0, 0);
  int k2 = F().operator++('0');
  // expected-error@-5 {{overloaded 'operator++' must be a unary or binary operator}}
  // expected-note@-3 {{in instantiation of function template specialization 'PR14995::F::operator++<int, int>' requested here}}
  // expected-error@-4 {{no matching member function for call to 'operator++'}}
  // expected-note@-8 {{candidate template ignored: substitution failure}}
  // expected-error@-9 {{parameter of overloaded post-increment operator must have type 'int' (not 'char')}}
  // expected-note@-6 {{in instantiation of function template specialization 'PR14995::F::operator++<char>' requested here}}
  // expected-error@-7 {{no matching member function for call to 'operator++'}}
  // expected-note@-12 {{candidate template ignored: substitution failure}}
} // namespace PR14995

namespace ConversionVersusTemplateOrdering {
  struct A {
    operator short() = delete;
    template <typename T> operator T();
  } a;
  struct B {
    template <typename T> operator T();
    operator short() = delete;
  } b;
  int x = a;
  int y = b;
}

namespace NoADLForMemberOnlyOperators {
  template<typename T> struct A { typename T::error e; }; // expected-error {{type 'char' cannot be used prior to '::'}}
  template<typename T> struct B { int n; };

  void f(B<A<void> > b1, B<A<int> > b2, B<A<char> > b3) {
    b1 = b1; // ok, does not instantiate A<void>.
    (void)b1->n; // expected-error {{is not a pointer}}
    b2[3]; // expected-error {{does not provide a subscript}}
    b3 / 0; // expected-note {{in instantiation of}} expected-error {{invalid operands to}}
  }
}


namespace PR27027 {
  template <class T> void operator+(T, T) = delete; // expected-note 4 {{candidate}}
  template <class T> void operator+(T) = delete; // expected-note 4 {{candidate}}

  struct A {} a_global;
  void f() {
    A a;
    +a; // expected-error {{overload resolution selected deleted operator '+'}}
    a + a; // expected-error {{overload resolution selected deleted operator '+'}}
    bool operator+(A);
    extern bool operator+(A, A);
    +a; // OK
    a + a;
  }
  bool test_global_1 = +a_global; // expected-error {{overload resolution selected deleted operator '+'}}
  bool test_global_2 = a_global + a_global; // expected-error {{overload resolution selected deleted operator '+'}}
}

namespace LateADLInNonDependentExpressions {
  struct A {};
  struct B : A {};
  int &operator+(A, A);
  int &operator!(A);
  int &operator+=(A, A);
  int &operator<<(A, A);
  int &operator++(A);
  int &operator++(A, int);
  int &operator->*(A, A);

  template<typename T> void f() {
    // An instantiation-dependent value of type B.
    // These are all non-dependent operator calls of type int&.
#define idB ((void()), B())
    int &a = idB + idB,
        &b = !idB,
        &c = idB += idB,
        &d = idB << idB,
        &e = ++idB,
        &f = idB++,
        &g = idB ->* idB;
  }

  // These should not be found by ADL in the template instantiation.
  float &operator+(B, B);
  float &operator!(B);
  float &operator+=(B, B);
  float &operator<<(B, B);
  float &operator++(B);
  float &operator++(B, int);
  float &operator->*(B, B);
  template void f<int>();
}

namespace test {
namespace A {
template<typename T> T f(T t) {
  T operator+(T, T);
  return t + t;
}
}
namespace B {
  struct X {};
}
void g(B::X x) { A::f(x); }
}

namespace GH78314 {

class a {
public:
  void operator--() = delete; // expected-note {{candidate function has been explicitly deleted}} \
                              // expected-note {{candidate function not viable: requires 0 arguments, but 1 was provided}}
  void operator--(int) = delete; // expected-note {{candidate function has been explicitly deleted}} \
                                 // expected-note {{candidate function not viable: requires 1 argument, but 0 were provided}}
};

class c {
  void operator--(this c) = delete; //precxx23-error {{explicit object parameters are incompatible with C++ standards before C++2b}} \
                                    // expected-note {{candidate function has been explicitly deleted}} \
                                    // expected-note {{candidate function not viable: requires 0 non-object arguments, but 1 was provided}}
  void operator--(this c, int) = delete; //precxx23-error {{explicit object parameters are incompatible with C++ standards before C++2b}} \
                                         // expected-note {{candidate function has been explicitly deleted}} \
                                         // expected-note {{candidate function not viable: requires 1 non-object argument, but 0 were provided}}
};

void foo() {
  a aa;
  --aa; // expected-error {{overload resolution selected deleted operator '--'}}
  aa--; // expected-error {{overload resolution selected deleted operator '--'}}

  c cc; 
  --cc; // expected-error {{overload resolution selected deleted operator '--'}}
  cc--; // expected-error {{overload resolution selected deleted operator '--'}}
}

class b {
  void operator++() = delete; // expected-note {{candidate function has been explicitly deleted}}
  template <class> void operator++(int) { // expected-note {{function template not viable: requires 1 argument, but 0 were provided}}
    b bb;
    ++bb; // expected-error {{overload resolution selected deleted operator '++'}}
  }
};


}

#if __cplusplus >= 202002L
namespace nw{
  template<class T>
  concept AlwaysTrue=true;

  struct S{
    template<class T>
    void operator+(const T&)const{}

    template<AlwaysTrue T>
    int operator-(const T&)const{return 0;}

    template<AlwaysTrue T>
    int operator*(const T&)const{ // expected-note {{candidate function}}
      return 0;
    }
  };

  template<AlwaysTrue T>
  int operator+(const S&, const T&){return 0;}

  template<class T>
  void operator-(const S&, const T&){}

  template<AlwaysTrue T>
  int operator*(const S&, const T&){ // expected-note {{candidate function}}
    return 0;
  }

  void foo(){
    int a = S{} + 1;
    int b = S{} - 1;
    int c = S{} * 1; // expected-error {{use of overloaded operator '*' is ambiguous (with operand types 'S' and 'int')}}
  }
}
#endif
