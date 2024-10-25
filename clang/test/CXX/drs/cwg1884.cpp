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
// provides all (significant) kinds of entities, each named 'a' through 'k'.
// Then the .cpp file does MxN kind of testing, where it tests one kind of entity
// against every other kind.

//--- cwg1884_A.cppm
export module cwg1884_A;

export {
int a;
void b();
enum E { c };
namespace d {}
struct e;
class f;
template <typename>
class g;
template <typename>
void h(int);
template <typename, typename>
int i;
template <typename>
using j = int;
template <typename>
concept k = true;
} // export


//--- cwg1884.cpp
import cwg1884_A;

// int a;

void a();
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
enum Ea {
  a
  // since-cxx20-error@-1 {{redefinition of 'a'}}
  //   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
};
namespace a {} // #cwg1884-namespace-a
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
struct a;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-a {{previous definition is here}}
class a;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-a {{previous definition is here}}
template <typename>
class a;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
template <typename>
void a(int);
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
template <typename, typename>
int a;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
template <typename T>
int a<T, int>;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using a = int;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}
template <typename>
concept a = true;
// since-cxx20-error@-1 {{redefinition of 'a' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:42 {{previous definition is here}}


// void b();

int b;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
enum Eb {
  b
  // since-cxx20-error@-1 {{redefinition of 'b'}}
  //   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
};
namespace b {} // #cwg1884-namespace-b
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
struct b;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-b {{previous definition is here}}
class b;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-b {{previous definition is here}}
template <typename>
class b;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
template <typename>
void b(int); // #cwg1884-func-template-b
// @-1 OK, a non-corresponding overload
template <typename, typename>
int b;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-func-template-b {{previous definition is here}}
template <typename T>
int b<T, int>;
// since-cxx20-error@-1 {{no variable template matches specialization; did you mean to use 'b' as function template instead?}}
template <typename>
using b = int;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:43 {{previous definition is here}}
template <typename>
concept b = true;
// since-cxx20-error@-1 {{redefinition of 'b' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-func-template-b {{previous definition is here}}


// enum E { c };

int c;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
void c();
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
namespace c {} // #cwg1884-namespace-c
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
struct c;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-c {{previous definition is here}}
class c;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-c {{previous definition is here}}
template <typename>
class c;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
template <typename>
void c(int);
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
template <typename, typename>
int c;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
template <typename T>
int c<T, int>;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using c = int;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}
template <typename>
concept c = true;
// since-cxx20-error@-1 {{redefinition of 'c' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:44 {{previous definition is here}}


// namespace d {};

int d;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
void d();
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
enum Ed {
  d
  // since-cxx20-error@-1 {{redefinition of 'd'}}
  //   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
};
struct d;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
class d;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
template <typename>
class d;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
template <typename>
void d(int);
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
template <typename, typename>
int d;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
template <typename T>
int d<T, int>;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using d = int;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}
template <typename>
concept d = true;
// since-cxx20-error@-1 {{redefinition of 'd' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:45 {{previous definition is here}}


// struct e;

int e; // #cwg1884-int-e
// @-1 OK, types and variables do not correspond
void e();
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
enum Ee {
  e
  // since-cxx20-error@-1 {{redefinition of 'e'}}
  //   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
};
namespace e {} // #cwg1884-namespace-e
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
class e;
// since-cxx20-error@-1 {{declaration of 'e' in the global module follows declaration in module cwg1884_A}}
//   since-cxx20-note@cwg1884_A.cppm:46 {{previous declaration is here}}
template <typename>
class e;
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
template <typename>
void e(int);
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
template <typename, typename>
int e;
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
template <typename T>
int e<T, int>;
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using e = int;
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}
template <typename>
concept e = true;
// since-cxx20-error@-1 {{redefinition of 'e' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-e {{previous definition is here}}


// class f;

int f; // #cwg1884-int-f
// @-1 OK, types and variables do not correspond
void f();
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
enum Ef {
  f
  // since-cxx20-error@-1 {{redefinition of 'f'}}
  //   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
};
namespace f {} // #cwg1884-namespace-f
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
struct f;
// since-cxx20-error@-1 {{declaration of 'f' in the global module follows declaration in module cwg1884_A}}
//   since-cxx20-note@cwg1884_A.cppm:47 {{previous declaration is here}}
template <typename>
class f;
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
template <typename>
void f(int);
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
template <typename, typename>
int f;
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
template <typename T>
int f<T, int>;
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
// since-cxx20-error@-3 {{expected ';' after top level declarator}}
template <typename>
using f = int;
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}
template <typename>
concept f = true;
// since-cxx20-error@-1 {{redefinition of 'f' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-int-f {{previous definition is here}}


// template <typename>
// class g;

int g;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
void g();
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
enum Eg {
  g
  // since-cxx20-error@-1 {{redefinition of 'g'}}
  //   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
};
namespace g {}
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
struct g;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
class g;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
template <typename>
void g(int);
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
template <typename, typename>
int g;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
template <typename T>
int g<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
using g = int;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}
template <typename>
concept g = true;
// since-cxx20-error@-1 {{redefinition of 'g' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:49 {{previous definition is here}}


// template <typename>
// void h(int);

int h;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
void h(); // #cwg1884-function-f
// @-1 OK, a non-corresponding overload
enum Eh {
  h
  // FIXME-since-cxx20-error@-1 {{redefinition of 'h'}}
  //   FIXME-since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
};
namespace h {} // #cwg1884-namespace-h
// FIXME-since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   FIXME-since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
struct h;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-h {{previous definition is here}}
class h;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-h {{previous definition is here}}
template <typename>
class h;
// FIXME-since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   FIXME-since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
template <typename, typename>
int h;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-namespace-h {{previous definition is here}}
template <typename T>
int h<T, int>;
// since-cxx20-error@-1 {{no variable template matches specialization; did you mean to use 'h' as function template instead?}}
template <typename>
using h = int;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:51 {{previous definition is here}}
template <typename>
concept h = true;
// since-cxx20-error@-1 {{redefinition of 'h' as different kind of symbol}}
//   since-cxx20-note@#cwg1884-function-f {{previous definition is here}}


// template <typename, typename>
// int i;

int i;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
void i();
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
enum Ei {
  i
  // since-cxx20-error@-1 {{redefinition of 'i'}}
  //   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
};
namespace i {}
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
struct i;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
class i;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
template <typename>
class i;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
template <typename>
void i(int);
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
template <typename T>
int i<T, int>; // OK, partial specialization
template <typename>
using i = int;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}
template <typename>
concept i = true;
// since-cxx20-error@-1 {{redefinition of 'i' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:53 {{previous definition is here}}


// template <typename>
// using j = int;

int j;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
void j();
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
enum Ej {
  j
  // since-cxx20-error@-1 {{redefinition of 'j'}}
  //   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
};
namespace j {}
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
struct j;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
class j;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
template <typename>
class j;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
template <typename>
void j(int);
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
template <typename, typename>
int j;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}
template <typename T>
int j<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
concept j = true;
// since-cxx20-error@-1 {{redefinition of 'j' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:55 {{previous definition is here}}


// template <typename>
// concept k = true;

int k;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
void k();
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
enum Ek {
  k
  // since-cxx20-error@-1 {{redefinition of 'k'}}
  //   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
};
namespace k {}
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
struct k;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
class k;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
template <typename>
class k;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
template <typename>
void k(int);
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
template <typename, typename>
int k;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
template <typename T>
int k<T, int>;
// since-cxx20-error@-1 {{no variable template matches partial specialization}}
template <typename>
using k = int;
// since-cxx20-error@-1 {{redefinition of 'k' as different kind of symbol}}
//   since-cxx20-note@cwg1884_A.cppm:57 {{previous definition is here}}
