// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file --leading-lines %s %t
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg1884_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg1884_A.pcm
// RUN: %clang_cc1 -std=c++20 -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg1884.cpp -fmodule-file=cwg1884_A=%t/cwg1884_A.pcm
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg1884_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg1884_A.pcm
// RUN: %clang_cc1 -std=c++23 -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg1884.cpp -fmodule-file=cwg1884_A=%t/cwg1884_A.pcm
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg1884_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg1884_A.pcm
// RUN: %clang_cc1 -std=c++2c -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg1884.cpp -fmodule-file=cwg1884_A=%t/cwg1884_A.pcm

// cwg1884: partial
// Cases b11, e11, g3, g4 are problematic, but we handle the other 101 cases fine.

// _N4993_.[basic.link]/11:
// For any two declarations of an entity E:
//   — If one declares E to be a variable or function,
//     the other shall declare E as one of the same type.
//   — If one declares E to be an enumerator, the other shall do so.
//   — If one declares E to be a namespace, the other shall do so.
//   — If one declares E to be a type,
//     the other shall declare E to be a type of the same kind (9.2.9.5).
//   — If one declares E to be a class template,
//     the other shall do so with the same kind and an equivalent template-head (13.7.7.2).
//     [Note 5 : The declarations can supply different default template arguments. — end note]
//   — If one declares E to be a function template or a (partial specialization of a) variable template,
//     the other shall declare E to be one with an equivalent template-head and type.
//   — If one declares E to be an alias template,
//     the other shall declare E to be one with an equivalent template-head and defining-type-id.
//   — If one declares E to be a concept, the other shall do so.
// Types are compared after all adjustments of types (during which typedefs (9.2.4) are replaced by their definitions);
// declarations for an array object can specify array types that differ by the presence or absence of a major array bound (9.3.4.5).
// No diagnostic is required if neither declaration is reachable from the other.

// The structure of the test is the following. First, module cwg1884_A
// provides all (significant) kinds of entities, each named 'a' through 'h', and copies of them.
// Then the .cpp file does MxN kind of testing, where it tests one kind of entity against every other kind.

//--- cwg1884_A.cppm
export module cwg1884_A;

export {
int a1;
int a2;
int a3;
int a4;
int a5;
int a6;
int a7;
int a8;
int a9;
int a10;
int a11;
void b1();
void b2();
void b3();
void b4();
void b5();
void b6();
void b7();
void b8();
void b9();
void b10();
void b11();
enum E {
  c1,
  c2, 
  c3,
  c4,
  c5,
  c6,
  c7,
  c8,
  c9,
  c10
};
namespace d1 {}
namespace d2 {}
namespace d3 {}
namespace d4 {}
namespace d5 {}
namespace d6 {}
namespace d7 {}
namespace d8 {}
namespace d9 {}
namespace d10 {}
struct e1;
struct e2;
struct e3;
struct e4;
struct e5;
struct e6;
struct e7;
struct e8;
struct e9;
struct e10;
struct e11;
struct e12;
struct e13;
template <typename>
class f1;
template <typename>
class f2;
template <typename>
class f3;
template <typename>
class f4;
template <typename>
class f5;
template <typename>
class f6;
template <typename>
class f7;
template <typename>
class f8;
template <typename>
class f9;
template <typename>
class f10;
template <typename>
class f11;
template <typename>
void g1(int);
template <typename>
void g2(int);
template <typename>
void g3(int);
template <typename>
void g4(int);
template <typename>
void g5(int);
template <typename>
void g6(int);
template <typename>
void g7(int);
template <typename>
void g8(int);
template <typename>
void g9(int);
template <typename>
void g10(int);
template <typename, typename>
int h1;
template <typename, typename>
int h2;
template <typename, typename>
int h3;
template <typename, typename>
int h4;
template <typename, typename>
int h5;
template <typename, typename>
int h6;
template <typename, typename>
int h7;
template <typename, typename>
int h8;
template <typename, typename>
int h9;
template <typename, typename>
int h10;
template <typename>
using i1 = int;
template <typename>
using i2 = int;
template <typename>
using i3 = int;
template <typename>
using i4 = int;
template <typename>
using i5 = int;
template <typename>
using i6 = int;
template <typename>
using i7 = int;
template <typename>
using i8 = int;
template <typename>
using i9 = int;
template <typename>
using i10 = int;
template <typename>
using i11 = int;
template <typename>
concept j1 = true;
template <typename>
concept j2 = true;
template <typename>
concept j3 = true;
template <typename>
concept j4 = true;
template <typename>
concept j5 = true;
template <typename>
concept j6 = true;
template <typename>
concept j7 = true;
template <typename>
concept j8 = true;
template <typename>
concept j9 = true;
template <typename>
concept j10 = true;
template <typename>
concept j11 = true;
} // export


//--- cwg1884.cpp
import cwg1884_A;

// FIXME: we don't diagnose several cases we should be. They are marked with MISSING prefix.

// Part A: matching against `int a;`
// ---------------------------------

void a1();
// since-cxx20-error@-1 {{redefinition of 'a1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
enum Ea {
  a2
  // since-cxx20-error@-1 {{redefinition of 'a2'}}
  //   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
};
namespace a3 {}
// since-cxx20-error@-1 {{redefinition of 'a3' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
struct a4;
// @-1 OK, types and variables do not correspond
template <typename>
class a5;
// since-cxx20-error@-1 {{redefinition of 'a5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:46 {{previous definition is here}}
template <typename>
void a6(int);
// since-cxx20-error@-1 {{redefinition of 'a6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:47 {{previous definition is here}}
template <typename, typename>
int a7;
// since-cxx20-error@-1 {{redefinition of 'a7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:48 {{previous definition is here}}
template <typename T>
int a8<T, int>;
// since-cxx20-error@-1 {{redefinition of 'a8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using a9 = int;
// since-cxx20-error@-1 {{redefinition of 'a9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:50 {{previous definition is here}}
template <typename>
concept a10 = true;
// since-cxx20-error@-1 {{redefinition of 'a10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
// For variables, type has to match as well.
long a11;
// since-cxx20-error@-1 {{redefinition of 'a11' with a different type: 'long' vs 'int'}}
//   since-cxx20-note@cwg1884_A.cppm:52 {{previous definition is here}}


// Part B: matching against `void b();`
// ------------------------------------

int b1;
// since-cxx20-error@-1 {{redefinition of 'b1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
enum Eb {
  b2
  // since-cxx20-error@-1 {{redefinition of 'b2'}}
  //   since-cxx20-note@cwg1884_A.cppm:54 {{previous definition is here}}
};
namespace b3 {} // #cwg1884-namespace-b
// since-cxx20-error@-1 {{redefinition of 'b3' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
struct b4;
// @-1 OK, types and functions do not correspond
template <typename>
class b5;
// since-cxx20-error@-1 {{redefinition of 'b5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
template <typename>
void b6(int);
// @-1 OK, a non-corresponding overload
template <typename, typename>
int b7;
// since-cxx20-error@-1 {{redefinition of 'b7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:59 {{previous definition is here}}
template <typename T>
int b8<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
using b9 = int;
// since-cxx20-error@-1 {{redefinition of 'b9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:61 {{previous definition is here}}
template <typename>
concept b10 = true;
// since-cxx20-error@-1 {{redefinition of 'b10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:62 {{previous definition is here}}
// For functions, type has to match as well.
// FIXME: we should be loud and clear here about type mismatch, like we do in `a11` case.
int b11();
// since-cxx20-error@-1 {{declaration of 'b11' in the global module follows declaration in module cwg1884_A}}
//   since-cxx20-note@cwg1884_A.cppm:63 {{previous declaration is here}}


// Part C: matching against `enum E { c };`
// ----------------------------------------

int c1;
// since-cxx20-error@-1 {{redefinition of 'c1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:65 {{previous definition is here}}
void c2();
// since-cxx20-error@-1 {{redefinition of 'c2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:66 {{previous definition is here}}
namespace c3 {}
// since-cxx20-error@-1 {{redefinition of 'c3' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:67 {{previous definition is here}}
struct c4;
// @-1 OK, types and enumerators do not correspond
template <typename>
class c5;
// since-cxx20-error@-1 {{redefinition of 'c5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:69 {{previous definition is here}}
template <typename>
void c6(int);
// since-cxx20-error@-1 {{redefinition of 'c6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:70 {{previous definition is here}}
template <typename, typename>
int c7;
// since-cxx20-error@-1 {{redefinition of 'c7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:71 {{previous definition is here}}
template <typename T>
int c8<T, int>;
// since-cxx20-error@-1 {{redefinition of 'c8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:72 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using c9 = int;
// since-cxx20-error@-1 {{redefinition of 'c9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:73 {{previous definition is here}}
template <typename>
concept c10 = true;
// since-cxx20-error@-1 {{redefinition of 'c10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:74 {{previous definition is here}}


// Part D: matching against `namespace d {};`
// ------------------------------------------

int d1;
// since-cxx20-error@-1 {{redefinition of 'd1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:76 {{previous definition is here}}
void d2();
// since-cxx20-error@-1 {{redefinition of 'd2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:77 {{previous definition is here}}
enum Ed {
  d3
  // since-cxx20-error@-1 {{redefinition of 'd3'}}
  //   since-cxx20-note@cwg1884_A.cppm:78 {{previous definition is here}}
};
struct d4;
// since-cxx20-error@-1 {{redefinition of 'd4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:79 {{previous definition is here}}
template <typename>
class d5;
// since-cxx20-error@-1 {{redefinition of 'd5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:80 {{previous definition is here}}
template <typename>
void d6(int);
// since-cxx20-error@-1 {{redefinition of 'd6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:81 {{previous definition is here}}
template <typename, typename>
int d7;
// since-cxx20-error@-1 {{redefinition of 'd7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:82 {{previous definition is here}}
template <typename T>
int d8<T, int>;
// since-cxx20-error@-1 {{redefinition of 'd8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:83 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using d9 = int;
// since-cxx20-error@-1 {{redefinition of 'd9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:84 {{previous definition is here}}
template <typename>
concept d10 = true;
// since-cxx20-error@-1 {{redefinition of 'd10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:85 {{previous definition is here}}


// Part E: matching against `struct e;`
// ------------------------------------

int e1;
// @-1 OK, types and variables do not correspond
void e2();
// @-1 OK, types and functions do not correspond
enum Ee {
  e3
  // @-1 OK, types and enumerators do not correspond
};
namespace e4 {}
// since-cxx20-error@-1 {{redefinition of 'e4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:89 {{previous definition is here}}
template <typename>
class e5;
// since-cxx20-error@-1 {{redefinition of 'e5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:90 {{previous definition is here}}
template <typename>
void e6(int);
// @-1 OK, types and function templates do not correspond
template <typename, typename>
int e7;
// since-cxx20-error@-1 {{redefinition of 'e7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:92 {{previous definition is here}}
template <typename T>
int e8<T, int>;
// since-cxx20-error@-1 {{redefinition of 'e8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:93 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using e9 = int;
// since-cxx20-error@-1 {{redefinition of 'e9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:94 {{previous definition is here}}
template <typename>
concept e10 = true;
// since-cxx20-error@-1 {{redefinition of 'e10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:95 {{previous definition is here}}
// FIXME: the following forward declaration is well-formed.
//        Agreement on 'struct' vs 'class' is not required per [dcl.type.elab]/7.
class e11;
// since-cxx20-error@-1 {{declaration of 'e11' in the global module follows declaration in module cwg1884_A}}
//   since-cxx20-note@cwg1884_A.cppm:96 {{previous declaration is here}}
union e12;
// since-cxx20-error@-1 {{use of 'e12' with tag type that does not match previous declaration}}
//   since-cxx20-note@cwg1884_A.cppm:97 {{previous use is here}}
// since-cxx20-error@-3 {{declaration of 'e12' in the global module follows declaration in module cwg1884_A}}
//   since-cxx20-note@cwg1884_A.cppm:97 {{previous declaration is here}}
enum e13 {};
// since-cxx20-error@-1 {{use of 'e13' with tag type that does not match previous declaration}}
//   since-cxx20-note@cwg1884_A.cppm:98 {{previous use is here}}


// Part F: matching against `template <typename> class f;`
// -------------------------------------------------------

int f1;
// since-cxx20-error@-1 {{redefinition of 'f1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:100 {{previous definition is here}}
void f2();
// since-cxx20-error@-1 {{redefinition of 'f2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:102 {{previous definition is here}}
enum Ef {
  f3
  // since-cxx20-error@-1 {{redefinition of 'f3'}}
  //   since-cxx20-note@cwg1884_A.cppm:104 {{previous definition is here}}
};
namespace f4 {}
// since-cxx20-error@-1 {{redefinition of 'f4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:106 {{previous definition is here}}
struct f5;
// since-cxx20-error@-1 {{redefinition of 'f5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:108 {{previous definition is here}}
template <typename>
void f6(int);
// since-cxx20-error@-1 {{redefinition of 'f6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:110 {{previous definition is here}}
template <typename, typename>
int f7;
// since-cxx20-error@-1 {{redefinition of 'f7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:112 {{previous definition is here}}
template <typename T>
int f8<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
using f9 = int;
// since-cxx20-error@-1 {{redefinition of 'f9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:116 {{previous definition is here}}
template <typename>
concept f10 = true;
// since-cxx20-error@-1 {{redefinition of 'f10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:118 {{previous definition is here}}


// Part G: matching against `template <typename> void g(int);`
// -----------------------------------------------------------

int g1;
// since-cxx20-error@-1 {{redefinition of 'g1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:122 {{previous definition is here}}
void g2();
// @-1 OK, a non-corresponding overload
enum Eg {
  g3
  // MISSING-since-cxx20-error@-1 {{redefinition of 'g3'}}
  //   MISSING-since-cxx20-note@cwg1884_A.cppm:126 {{previous definition is here}}
};
namespace g4 {}
// MISSING-since-cxx20-error@-1 {{redefinition of 'g4' as different kind of symbol}}
//   MISSING-since-cxx20-note@cwg1884_A.cppm:128 {{previous definition is here}}
struct g5;
// @-1 OK, types and function templates do not correspond
template <typename>
class g6;
// since-cxx20-error@-1 {{redefinition of 'g6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:132 {{previous definition is here}}
template <typename, typename>
int g7;
// since-cxx20-error@-1 {{redefinition of 'g7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:134 {{previous definition is here}}
template <typename T>
int g8<T, int>;
// since-cxx20-error@-1 {{no variable template matches specialization; did you mean to use 'g8' as function template instead?}}
template <typename>
using g9 = int;
// since-cxx20-error@-1 {{redefinition of 'g9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:138 {{previous definition is here}}
template <typename>
concept g10 = true;
// since-cxx20-error@-1 {{redefinition of 'g10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:140 {{previous definition is here}}


// Part H: matching against `template <typename, typename> int h;`
// ---------------------------------------------------------------

int h1;
// since-cxx20-error@-1 {{redefinition of 'h1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:142 {{previous definition is here}}
void h2();
// since-cxx20-error@-1 {{redefinition of 'h2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:144 {{previous definition is here}}
enum Eh {
  h3
  // since-cxx20-error@-1 {{redefinition of 'h3'}}
  //   since-cxx20-note@cwg1884_A.cppm:146 {{previous definition is here}}
};
namespace h4 {}
// since-cxx20-error@-1 {{redefinition of 'h4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:148 {{previous definition is here}}
struct h5;
// since-cxx20-error@-1 {{redefinition of 'h5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:150 {{previous definition is here}}
template <typename>
class h6;
// since-cxx20-error@-1 {{redefinition of 'h6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:152 {{previous definition is here}}
template <typename>
void h7(int);
// since-cxx20-error@-1 {{redefinition of 'h7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:154 {{previous definition is here}}
template <typename T>
int h8<T, int>;
// @-1 OK, partial specialization
template <typename>
using h9 = int;
// since-cxx20-error@-1 {{redefinition of 'h9' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:158 {{previous definition is here}}
template <typename>
concept h10 = true;
// since-cxx20-error@-1 {{redefinition of 'h10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:160 {{previous definition is here}}


// Part I: matching against `template <typename> using i = int;`
// -------------------------------------------------------------

int i1;
// since-cxx20-error@-1 {{redefinition of 'i1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:162 {{previous definition is here}}
void i2();
// since-cxx20-error@-1 {{redefinition of 'i2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:164 {{previous definition is here}}
enum Ei {
  i3
  // since-cxx20-error@-1 {{redefinition of 'i3'}}
  //   since-cxx20-note@cwg1884_A.cppm:166 {{previous definition is here}}
};
namespace i4 {}
// since-cxx20-error@-1 {{redefinition of 'i4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:168 {{previous definition is here}}
struct i5;
// since-cxx20-error@-1 {{redefinition of 'i5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:170 {{previous definition is here}}
template <typename>
class i6;
// since-cxx20-error@-1 {{redefinition of 'i6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:172 {{previous definition is here}}
template <typename>
void i7(int);
// since-cxx20-error@-1 {{redefinition of 'i7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:174 {{previous definition is here}}
template <typename, typename>
int i8;
// since-cxx20-error@-1 {{redefinition of 'i8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:176 {{previous definition is here}}
template <typename T>
int i9<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
concept i10 = true;
// since-cxx20-error@-1 {{redefinition of 'i10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:180 {{previous definition is here}}


// Part J: matching against `template <typename> concept j = true;`
// ----------------------------------------------------------------

int j1;
// since-cxx20-error@-1 {{redefinition of 'j1' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:184 {{previous definition is here}}
void j2();
// since-cxx20-error@-1 {{redefinition of 'j2' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:186 {{previous definition is here}}
enum Ej {
  j3
  // since-cxx20-error@-1 {{redefinition of 'j3'}}
  //   since-cxx20-note@cwg1884_A.cppm:188 {{previous definition is here}}
};
namespace j4 {}
// since-cxx20-error@-1 {{redefinition of 'j4' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:190 {{previous definition is here}}
struct j5;
// since-cxx20-error@-1 {{redefinition of 'j5' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:192 {{previous definition is here}}
template <typename>
class j6;
// since-cxx20-error@-1 {{redefinition of 'j6' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:194 {{previous definition is here}}
template <typename>
void j7(int);
// since-cxx20-error@-1 {{redefinition of 'j7' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:196 {{previous definition is here}}
template <typename, typename>
int j8;
// since-cxx20-error@-1 {{redefinition of 'j8' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:198 {{previous definition is here}}
template <typename T>
int j9<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
using j10 = int;
// since-cxx20-error@-1 {{redefinition of 'j10' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:202 {{previous definition is here}}
