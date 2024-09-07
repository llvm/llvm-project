// RUN: %clang_cc1 -std=c++17 -fsyntax-only %s -verify=expected,pre20 -Wno-c++2a-extensions
// RUN: %clang_cc1 -std=c++2a -fsyntax-only %s -verify=expected,post20

template <bool b, auto val> struct enable_ifv {};

template <auto val> struct enable_ifv<true, val> {
  static constexpr auto value = val;
};

template <typename T1, typename T2> struct is_same {
  static constexpr bool value = false;
};

template <typename T> struct is_same<T, T> {
  static constexpr bool value = true;
};

namespace special_cases
{

template<int a>
struct A {
// pre20-note@-1+ {{candidate constructor}}
  explicit(1 << a)
// expected-note@-1 {{negative shift count -1}}
// expected-error@-2 {{explicit specifier argument is not a constant expression}}
  A(int);
};

A<-1> a(0);
// pre20-error@-1 {{no matching constructor}}
// post20-error@-2 {{excess elements in struct initializer}}
// expected-note@-3 {{in instantiation of template class}}

template<int a>
struct B {
  explicit(b)
  // expected-error@-1 {{use of undeclared identifier}}
  B(int);
};

template<int a>
struct B1 {
  explicit(a +)
  // expected-error@-1 {{expected expression}}
  B1(int);
};

struct B2 {
  explicit(false) explicit
  B2(int);
  // expected-error@-2 {{duplicate 'explicit' declaration specifier}}
};

template<int a>
  struct C {
  // expected-note@-1 {{candidate constructor}} expected-note@-1 {{candidate constructor}}
  // expected-note@-2 {{candidate constructor}} expected-note@-2 {{candidate constructor}}
  explicit(a == 0)
C(int), // expected-note 2{{not a candidate}}
C(double); // expected-note 2{{not a candidate}}
};

C<0> c0 = 0.0; // expected-error {{no viable conversion}}
C<0> c1 = 0; // expected-error {{no viable conversion}}
C<1> c2 = 0.0;
C<1> c3 = 0;

explicit(false) void f(int);// expected-error {{'explicit' can only be specified inside the class definition}}

struct D {
  explicit(false) void f(int);// expected-error {{'explicit' can only be applied to a constructor or conversion function}}
};

template <typename T> struct E {
  // expected-note@-1+ {{candidate constructor}}
  explicit((T{}, false))
  // expected-error@-1 {{cannot create object of function type 'void ()'}}
  E(int);
};

E<void ()> e = 1;
// expected-error@-1 {{no viable conversion}}
// expected-note@-2 {{in instantiation of}}

}

namespace trailing_object {

template<bool b>
struct B {
  explicit(b) B(int) {}
};

template<bool b>
struct A : B<b> {
  explicit(b) A(int) : B<b>(0) {}
};

A<true> a(0);

}

namespace constructor1 {

template<bool b>
  struct A {
    // expected-note@-1+ {{candidate constructor}}
    // expected-note@-2+ {{candidate function}}
    explicit(b) A(int, int = 0); // expected-note {{not a candidate}}
  // expected-note@-1+ {{explicit constructor declared here}}
};

template<bool b>
A<b>::A(int, int) {}

void f()
{
A<true> a0 = 0; // expected-error {{no viable conversion}}
A<true> a1( 0);
A<true> && a2 = 0;// expected-error {{could not bind}}
A<true> && a3( 0);// expected-error {{could not bind}}
A<true> a4{ 0};
A<true> && a5 = { 0};// expected-error {{chosen constructor is explicit}}
A<true> && a6{ 0};
A<true> a7 = { 0}; // expected-error {{chosen constructor is explicit in copy-initialization}}

a0 = 0; // expected-error {{no viable overloaded '='}}
a1 = { 0}; // expected-error {{no viable overloaded '='}}
a2 = A<true>( 0);
a3 = A<true>{ 0};

A<false> c0 =  ((short)0);
A<false> c1( ((short)0));
A<false> && c2 =  ((short)0);
A<false> && c3( ((short)0));
A<false> c4{ ((short)0)};
A<false> && c5 = { ((short)0)};
A<false> && c6{ ((short)0)};

A<true> d1( 0, 0);
A<true> d2{ 0, 0};
A<true> d3 = { 0, 0}; // expected-error {{chosen constructor is explicit in copy-initialization}}

d1 = { 0, 0}; // expected-error {{no viable overloaded '='}}
d2 = A<true>( 0, 0);
d3 = A<true>{ 0, 0};
}
}

namespace constructor2 {

template<bool a, typename T1>
struct A {
  // expected-note@-1 {{candidate constructor}} expected-note@-1 {{candidate constructor}}
  // expected-note@-2 {{candidate constructor}} expected-note@-2 {{candidate constructor}}
  template<typename T2>
  explicit(a ^ is_same<T1, T2>::value)
  A(T2) {}
  // expected-note@-1+ {{explicit constructor declared here}}
  // expected-note@-2+ {{not a candidate}}
};

A<true, int> a0 = 0.0; // expected-error {{no viable conversion}}
A<true, int> a1( 0.0);
A<true, int> && a2 = 0.0;// expected-error {{could not bind}}
A<true, int> && a3( 0.0);// expected-error {{could not bind}}
A<true, int> a4{ 0.0};
A<true, int> && a5 = { 0.0};// expected-error {{chosen constructor is explicit}}
A<true, int> && a6{ 0.0};
A<true, int> a7 = { 0.0}; // expected-error {{chosen constructor is explicit in copy-initialization}}

A<true, int> b0 = 0;
A<true, int> b1( 0);
A<true, int> && b2 = 0;
A<true, int> && b3( 0);
A<true, int> b4{ 0};
A<true, int> && b5 = { 0};
A<true, int> && b6{ 0};
A<true, int> b7 = { 0};

A<true, double> c0 = 0; // expected-error {{no viable conversion}}
A<true, double> c1( 0);
A<true, double> && c2 = 0;// expected-error {{could not bind}}
A<true, double> && c3( 0);// expected-error {{could not bind}}
A<true, double> c4{ 0};
A<true, double> && c5 = { 0};// expected-error {{chosen constructor is explicit}}
A<true, double> && c6{ 0};
A<true, double> c7 = { 0}; // expected-error {{chosen constructor is explicit in copy-initialization}}

}

namespace constructor_sfinae {

template<bool a>
struct A {
  // expected-note@-1+ {{candidate constructor}}
  template<typename T>
  explicit(enable_ifv<is_same<int, T>::value, a>::value)
  A(T) {}
  // expected-note@-1+ {{substitution failure}}
  // expected-note@-2 {{not a candidate}}
  // expected-note@-3 {{explicit constructor declared here}}
  template<typename T, bool c = true>
  explicit(enable_ifv<is_same<bool, T>::value, a>::value)
  A(T) {}
  // expected-note@-1+ {{substitution failure}}
  // expected-note@-2 {{not a candidate}}
  // expected-note@-3 {{explicit constructor declared here}}
};

A<true> a0 = 0.0; // expected-error {{no viable conversion}}
A<true> a1( 0.0); // expected-error {{no matching constructor}}
A<true> a4{ 0.0}; // expected-error {{no matching constructor}}
A<true> a7 = { 0.0}; // expected-error {{no matching constructor}}

A<true> b0 = 0; // expected-error {{no viable conversion}}
A<true> b1( 0);
A<true> b4{ 0};
A<true> b7 = { 0}; // expected-error {{chosen constructor is explicit}}

A<false> c0 = 0;
A<false> c1( 0);
A<false> c4{ 0};
A<false> c7 = { 0};

A<true> d0 = true; // expected-error {{no viable conversion}}
A<true> d1( true);
A<true> d4{ true};
A<true> d7 = { true}; // expected-error {{chosen constructor is explicit}}

}

namespace conversion {

template<bool a>
struct A {
  explicit(a) operator int (); // expected-note+ {{not a candidate}}
};

template<bool a>
A<a>::operator int() {
  return 0;
}

A<true> A_true;
A<false> A_false;

int ai0 = A<true>(); // expected-error {{no viable conversion}}
const int& ai1 = A<true>(); // expected-error {{no viable conversion}}
int&& ai3 = A<true>(); // expected-error {{no viable conversion}}
int ai4 = A_true; // expected-error {{no viable conversion}}
const int& ai5 = A_true; // expected-error {{no viable conversion}}

int ai01 = {A<true>()}; // expected-error {{no viable conversion}}
const int& ai11 = {A<true>()}; // expected-error {{no viable conversion}}
int&& ai31 = {A<true>()}; // expected-error {{no viable conversion}}
int ai41 = {A_true}; // expected-error {{no viable conversion}}
const int& ai51 = {A_true}; // expected-error {{no viable conversion}}

int ae0(A<true>());
const int& ae1(A<true>());
int&& ae3(A<true>());
int ae4(A_true);
const int& ae5(A_true);

int bi0 = A<false>();
const int& bi1 = A<false>();
int&& bi3 = A<false>();
int bi4 = A_false;
const int& bi5 = A_false;

int bi01 = {A<false>()};
const int& bi11 = {A<false>()};
int&& bi31 = {A<false>()};
int bi41 = {A_false};
const int& bi51 = {A_false};

int be0(A<true>());
const int& be1(A<true>());
int&& be3(A<true>());
int be4(A_true);
const int& be5(A_true);

}

namespace conversion2 {

struct B {};
// expected-note@-1+ {{candidate constructor}}
template<bool a>
struct A {
  template<typename T2>
  explicit(enable_ifv<is_same<B, T2>::value, a>::value)
  operator T2() { return T2(); };
  // expected-note@-1+ {{substitution failure}}
  // expected-note@-2+ {{not a candidate}}
};

A<false> A_false;
A<true> A_true;

int ai0 = A<true>(); // expected-error {{no viable conversion}}
const int& ai1 = A<true>(); // expected-error {{no viable conversion}}
int&& ai3 = A<true>(); // expected-error {{no viable conversion}}
int ai4 = A_false; // expected-error {{no viable conversion}}
const int& ai5 = A_false; // expected-error {{no viable conversion}}

int ae0{A<true>()};  // expected-error {{no viable conversion}}
const int& ae1{A<true>()};  // expected-error {{no viable conversion}}
int&& ae3{A<true>()};  // expected-error {{no viable conversion}}
int ae4{A_true};  // expected-error {{no viable conversion}}
const int& ae5{A_true};  // expected-error {{no viable conversion}}

int ap0((A<true>()));  // expected-error {{no viable conversion}}
const int& ap1((A<true>()));  // expected-error {{no viable conversion}}
int&& ap3((A<true>()));  // expected-error {{no viable conversion}}
int ap4(A_true);  // expected-error {{no viable conversion}}
const int& ap5(A_true);  // expected-error {{no viable conversion}}

B b0 = A<true>(); // expected-error {{no viable conversion}}
const B & b1 = A<true>(); // expected-error {{no viable conversion}}
B && b3 = A<true>(); // expected-error {{no viable conversion}}
B b4 = A_true; // expected-error {{no viable conversion}}
const B & b5 = A_true; // expected-error {{no viable conversion}}

B be0(A<true>());
const B& be1(A<true>());
B&& be3(A<true>());
B be4(A_true);
const B& be5(A_true);

B c0 = A<false>();
const B & c1 = A<false>();
B && c3 = A<false>();
B c4 = A_false;
const B & c5 = A_false;

}

namespace parameter_pack {

template<typename T>
struct A {
  // expected-note@-1+ {{candidate constructor}}
  // expected-note@-2+ {{candidate function}}
  template<typename ... Ts>
  explicit((is_same<T, Ts>::value && ...))
  A(Ts...);
  // expected-note@-1 {{not a candidate}}
  // expected-note@-2 {{explicit constructor}}
};

template<typename T>
template<typename ... Ts>
A<T>::A(Ts ...) {}

void f() {

A<int> a0 = 0; // expected-error {{no viable conversion}}
A<int> a1( 0, 1);
A<int> a2{ 0, 1};
A<int> a3 = { 0, 1}; // expected-error {{chosen constructor is explicit}}

a1 = 0; // expected-error {{no viable overloaded '='}}
a2 = { 0, 1}; // expected-error {{no viable overloaded '='}}

A<double> b0 = 0;
A<double> b1( 0, 1);
A<double> b2{ 0, 1};
A<double> b3 = { 0, 1};

b1 = 0;
b2 = { 0, 1};

}

}

namespace deduction_guide {

template<bool b>
struct B {};

B<true> b_true;
B<false> b_false;

template<typename T>
struct nondeduced
{
using type = T;
};

template<typename T1, typename T2, bool b>
struct A {
  // expected-note@-1+ {{candidate function}}
  // expected-note@-2+ {{implicit deduction guide}}
  explicit(false)
  A(typename nondeduced<T1>::type, typename nondeduced<T2>::type, typename nondeduced<B<b>>::type) {}
  // expected-note@-1+ {{candidate template ignored}}
  // expected-note@-2+ {{implicit deduction guide}}
};

template<typename T1, typename T2, bool b>
explicit(enable_ifv<is_same<T1, T2>::value, b>::value)
A(T1, T2, B<b>) -> A<T1, T2, b>;
// expected-note@-1+ {{explicit deduction guide declared here}}
// expected-note@-2+ {{candidate template ignored}}
void f() {

A a0( 0.0, 1, b_true); // expected-error {{no viable constructor or deduction guide}}
A a1{ 0.0, 1, b_true}; // expected-error {{no viable constructor or deduction guide}}
A a2 = { 0.0, 1, b_true}; // expected-error {{no viable constructor or deduction guide}}
auto a4 = A( 0.0, 1, b_true); // expected-error {{no viable constructor or deduction guide}}
auto a5 = A{ 0.0, 1, b_true}; // expected-error {{no viable constructor or deduction guide}}

A b0( 0, 1, b_true);
A b1{ 0, 1, b_true};
A b2 = { 0, 1, b_true}; // expected-error {{explicit deduction guide for copy-list-initialization}}
auto b4 = A( 0, 1, b_true);
auto b5 = A{ 0, 1, b_true};
b0 = { 0, 1, b_false}; // expected-error {{no viable overloaded '='}}

A c0( 0, 1, b_false);
A c1{ 0, 1, b_false};
A c2 = { 0, 1, b_false};
auto c4 = A( 0, 1, b_false);
auto c5 = A{ 0, 1, b_false};
c2 = { 0, 1, b_false};

}

}

namespace test8 {

template<bool b>
struct A {
  //expected-note@-1+ {{candidate function}}
  template<typename T1, typename T2>
  explicit(b)
  A(T1, T2) {}
  //expected-note@-1 {{explicit constructor declared here}}
};

template<typename T1, typename T2>
explicit(!is_same<T1, int>::value)
A(T1, T2) -> A<!is_same<int, T2>::value>;
// expected-note@-1+ {{explicit deduction guide declared here}}

template<bool b>
A<b> v();

void f() {

A a0( 0, 1);
A a1{ 0, 1};
A a2 = { 0, 1};
auto a4 = A( 0, 1);
auto a5 = A{ 0, 1};
auto a6(v<false>());
a6 = { 0, 1};

A b0( 0.0, 1);
A b1{ 0.0, 1};
A b2 = { 0.0, 1}; // expected-error {{explicit deduction guide for copy-list-initialization}}
auto b4 = A( 0.0, 1);
auto b5 = A{ 0.0, 1};

A c0( 0, 1.0);
A c1{ 0, 1.0};
A c2 = { 0, 1.0}; // expected-error {{chosen constructor is explicit}}
auto c4 = A( 0, 1.0);
auto c5 = A{ 0, 1.0};
auto c6(v<true>());
c0 = { 0, 1.0}; // expected-error {{no viable overloaded '='}}

A d0( 0.0, 1.0);
A d1{ 0.0, 1.0};
A d2 = { 0.0, 1.0};  // expected-error {{explicit deduction guide for copy-list-initialization}}
auto d4 = A( 0.0, 1.0);
auto d5 = A{ 0.0, 1.0};

}

}

namespace conversion3 {

template<bool b>
struct A {
  explicit(!b) operator int();
  explicit(b) operator bool();
};

template<bool b>
A<b>::operator bool() { return false; }

struct B {
  void f(int);
  void f(bool);
};

void f(A<true> a, B b) {
  b.f(a);
}

void f1(A<false> a, B b) {
  b.f(a);
}

// Taken from 12.3.2p2
class X { X(); };
class Y { }; // expected-note+ {{candidate constructor (the implicit}}

template<bool b>
struct Z {
  explicit(b) operator X() const;
  explicit(b) operator Y() const; // expected-note 2{{not a candidate}}
  explicit(b) operator int() const; // expected-note {{not a candidate}}
};

void testExplicit()
{
Z<true> z;
// 13.3.1.4p1 & 8.5p16:
Y y2 = z; // expected-error {{no viable conversion}}
Y y2b(z);
Y y3 = (Y)z;
Y y4 = Y(z);
Y y5 = static_cast<Y>(z);
// 13.3.1.5p1 & 8.5p16:
int i1 = (int)z;
int i2 = int(z);
int i3 = static_cast<int>(z);
int i4(z);
// 13.3.1.6p1 & 8.5.3p5:
const Y& y6 = z; // expected-error {{no viable conversion}}
const int& y7 = z; // expected-error {{no viable conversion}}
const Y& y8(z);
const int& y9(z);

// Y is an aggregate, so aggregate-initialization is performed and the
// conversion function is not considered.
const Y y10{z}; // expected-error {{excess elements}}
const Y& y11{z}; // expected-error {{excess elements}} expected-note {{in initialization of temporary}}
const int& y12{z};

// X is not an aggregate, so constructors are considered,
// per 13.3.3.1/4 & DR1467.
const X x1{z};
const X& x2{z};
}

struct tmp {};

template<typename T1>
struct C {
  template<typename T>
  explicit(!is_same<T1, T>::value)
  operator T(); // expected-note+ {{explicit conversion function is not a candidate}}
};

using Bool = C<bool>;
using Integral = C<int>;
using Unrelated = C<tmp>;

void testBool() {
Bool    b;
Integral n;
Unrelated u;

(void) (1 + b); // expected-error {{invalid operands to binary expression}}
(void) (1 + n);
(void) (1 + u); // expected-error {{invalid operands to binary expression}}

// 5.3.1p9:
(void) (!b);
(void) (!n);
(void) (!u);

// 5.14p1:
(void) (b && true);
(void) (n && true);
(void) (u && true);

// 5.15p1:
(void) (b || true);
(void) (n || true);
(void) (u || true);

// 5.16p1:
(void) (b ? 0 : 1);
(void) (n ? 0: 1);
(void) (u ? 0: 1);

// // 5.19p5:
// // TODO: After constexpr has been implemented

// 6.4p4:
if (b) {}
if (n) {}
if (u) {}

// 6.4.2p2:
switch (b) {} // expected-error {{statement requires expression of integer type}}
switch (n) {} // expected-error {{statement requires expression of integer type}}
switch (u) {} // expected-error {{statement requires expression of integer type}}

// 6.5.1:
while (b) {}
while (n) {}
while (u) {}

// 6.5.2p1:
do {} while (b);
do {} while (n);
do {} while (u);

// 6.5.3:
for (;b;) {}
for (;n;) {}
for (;u;) {}

// 13.3.1.5p1:
bool db1(b);
bool db2(n);
bool db3(u);
int di1(b);
int di2(n);
int di3(n);
const bool &direct_cr1(b);
const bool &direct_cr2(n);
const bool &direct_cr3(n);
const int &direct_cr4(b);
const int &direct_cr5(n);
const int &direct_cr6(n);
bool directList1{b};
bool directList2{n};
bool directList3{n};
int directList4{b};
int directList5{n};
int directList6{n};
const bool &directList_cr1{b};
const bool &directList_cr2{n};
const bool &directList_cr3{n};
const int &directList_cr4{b};
const int &directList_cr5{n};
const int &directList_cr6{n};
bool copy1 = b;
bool copy2 = n;// expected-error {{no viable conversion}}
bool copyu2 = u;// expected-error {{no viable conversion}}
int copy3 = b;// expected-error {{no viable conversion}}
int copy4 = n;
int copyu4 = u;// expected-error {{no viable conversion}}
const bool &copy5 = b;
const bool &copy6 = n;// expected-error {{no viable conversion}}
const bool &copyu6 = u;// expected-error {{no viable conversion}}
const int &copy7 = b;// expected-error {{no viable conversion}}
const int &copy8 = n;
const int &copyu8 = u;// expected-error {{no viable conversion}}
bool copyList1 = {b};
bool copyList2 = {n};// expected-error {{no viable conversion}}
bool copyListu2 = {u};// expected-error {{no viable conversion}}
int copyList3 = {b};// expected-error {{no viable conversion}}
int copyList4 = {n};
int copyListu4 = {u};// expected-error {{no viable conversion}}
const bool &copyList5 = {b};
const bool &copyList6 = {n};// expected-error {{no viable conversion}}
const bool &copyListu6 = {u};// expected-error {{no viable conversion}}
const int &copyList7 = {b};// expected-error {{no viable conversion}}
const int &copyList8 = {n};
const int &copyListu8 = {u};// expected-error {{no viable conversion}}
}

}

namespace deduction_guide2 {

template<typename T1 = int, typename T2 = int>
struct A {
  // expected-note@-1+ {{candidate template ignored}}
  // expected-note@-2+ {{implicit deduction guide}}
  explicit(!is_same<T1, T2>::value)
  A(T1 = 0, T2 = 0) {}
  // expected-note@-1 {{explicit constructor declared here}}
  // expected-note@-2 2{{explicit constructor is not a candidate}}
  // expected-note@-3 2{{implicit deduction guide declared}}
};

A a0 = 0;
A a1(0, 0);
A a2{0, 0};
A a3 = {0, 0};

A b0 = 0.0; // expected-error {{no viable constructor or deduction guide}}
A b1(0.0, 0.0);
A b2{0.0, 0.0};
A b3 = {0.0, 0.0};

A b4 = {0.0, 0}; // expected-error {{explicit constructor}}

template<typename T1, typename T2>
explicit A(T1, T2) -> A<T1, T2>;
// expected-note@-1+ {{explicit deduction guide}}

A c0 = 0;
A c1(0, 0);
A c2{0, 0};
A c3 = {0, 0};// expected-error {{explicit deduction guide}}

A d0 = 0.0; // expected-error {{no viable constructor or deduction guide}}
A d1(0, 0);
A d2{0, 0};
A d3 = {0.0, 0.0};// expected-error {{explicit deduction guide}}

}

namespace PR42980 {
using size_t = decltype(sizeof(0));

struct Str {// expected-note+ {{candidate constructor}}
  template <size_t N>
  explicit(N > 7)
  Str(char const (&str)[N]); // expected-note {{explicit constructor is not a candidate}}
};

template <size_t N>
Str::Str(char const(&str)[N]) { }

Str a = "short";
Str b = "not so short";// expected-error {{no viable conversion}}

}

namespace P1401 {

const int *ptr;

struct S {
  explicit(sizeof(char[2])) S(char); // expected-error {{explicit specifier argument evaluates to 2, which cannot be narrowed to type 'bool'}}
  explicit(ptr) S(long);             // expected-error {{conversion from 'const int *' to 'bool' is not allowed in a converted constant expression}}
  explicit(nullptr) S(int);          // expected-error {{conversion from 'std::nullptr_t' to 'bool' is not allowed in a converted constant expression}}
  explicit(42L) S(int, int);         // expected-error {{explicit specifier argument evaluates to 42, which cannot be narrowed to type 'bool'}}
  explicit(sizeof(char)) S();
  explicit(0) S(char, char);
  explicit(1L) S(char, char, char);
};
} // namespace P1401
