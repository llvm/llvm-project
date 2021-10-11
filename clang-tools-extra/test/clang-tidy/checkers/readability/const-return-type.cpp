// RUN: %check_clang_tidy -std=c++14 %s readability-const-return-type %t

//  p# = positive test
//  n# = negative test

namespace std {
template< class T >
struct add_cv { typedef const volatile T type; };

template< class T> struct add_const { typedef const T type; };

template< class T> struct add_volatile { typedef volatile T type; };
}

const int p1() {
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qualified at the top level, which may reduce code readability without improving const correctness
// CHECK-FIXES: int p1() {
  return 1;
}

const int p15();
// CHECK-FIXES: int p15();

template <typename T>
const int p31(T v) { return 2; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
// CHECK-FIXES: int p31(T v) { return 2; }

// We detect const-ness even without instantiating T.
template <typename T>
const T p32(T t) { return t; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const T' is 'const'-qual
// CHECK-FIXES: T p32(T t) { return t; }

// However, if the return type is itself a template instantiation, Clang does
// not consider it const-qualified without knowing `T`.
template <typename T>
typename std::add_const<T>::type n15(T v) { return v; }

template <bool B>
struct MyStruct {};

template <typename A>
class Klazz {
public:
  Klazz(A) {}
};

class Clazz {
 public:
  Clazz *const p2() {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'Clazz *const' is 'co
    // CHECK-FIXES: Clazz *p2() {
    return this;
  }

  Clazz *const p3();
  // CHECK-FIXES: Clazz *p3();

  const int p4() const {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const int' is 'const
    // CHECK-FIXES: int p4() const {
    return 4;
  }

  const Klazz<const int>* const p5() const;
  // CHECK-FIXES: const Klazz<const int>* p5() const;

  const Clazz operator++(int x) {  //  p12
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const Clazz' is 'const
  // CHECK-FIXES: Clazz operator++(int x) {
  }

  struct Strukt {
    int i;
  };

  const Strukt p6() {}
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const Strukt' i
  // CHECK-FIXES: Strukt p6() {}

  // No warning is emitted here, because this is only the declaration.  The
  // warning will be associated with the definition, below.
  const Strukt* const p7();
  // CHECK-FIXES: const Strukt* p7();

  // const-qualifier is the first `const` token, but not the first token.
  static const int p8() {}
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const int' is 'const'-
  // CHECK-FIXES: static int p8() {}

  static const Strukt p9() {}
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const Strukt' i
  // CHECK-FIXES: static Strukt p9() {}

  int n0() const { return 0; }
  const Klazz<const int>& n11(const Klazz<const int>) const;
};

Clazz *const Clazz::p3() {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'Clazz *const' is 'cons
  // CHECK-FIXES: Clazz *Clazz::p3() {
  return this;
}

const Klazz<const int>* const Clazz::p5() const {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const int> *
// CHECK-FIXES: const Klazz<const int>* Clazz::p5() const {}

const Clazz::Strukt* const Clazz::p7() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Clazz::Strukt *con
// CHECK-FIXES: const Clazz::Strukt* Clazz::p7() {}

Clazz *const p10();
// CHECK-FIXES: Clazz *p10();

Clazz *const p10() {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'Clazz *const' is 'cons
  // CHECK-FIXES: Clazz *p10() {
  return new Clazz();
}

const Clazz bar;
const Clazz *const p11() {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Clazz *const' is
  // CHECK-FIXES: const Clazz *p11() {
  return &bar;
}

const Klazz<const int> p12() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const int>'
// CHECK-FIXES: Klazz<const int> p12() {}

const Klazz<const Klazz<const int>> p33() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<
// CHECK-FIXES: Klazz<const Klazz<const int>> p33() {}

const Klazz<const int>* const p13() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const int> *
// CHECK-FIXES: const Klazz<const int>* p13() {}

const Klazz<const int>* const volatile p14() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const int> *
// CHECK-FIXES: const Klazz<const int>* volatile p14() {}

const MyStruct<0 < 1> p34() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const MyStruct<0 < 1>'
// CHECK-FIXES: MyStruct<0 < 1> p34() {}

MyStruct<0 < 1> const p35() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const MyStruct<0 < 1>'
// CHECK-FIXES: MyStruct<0 < 1> p35() {}

Klazz<MyStruct<0 < 1> const> const p36() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const MyStru
// CHECK-FIXES: Klazz<MyStruct<0 < 1> const> p36() {}

const Klazz<MyStruct<0 < 1> const> *const p37() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const MyStru
// CHECK-FIXES: const Klazz<MyStruct<0 < 1> const> *p37() {}

Klazz<const MyStruct<0 < 1>> const p38() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const MyStru
// CHECK-FIXES: Klazz<const MyStruct<0 < 1>> p38() {}

const Klazz<const MyStruct<0 < 1>> p39() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<
// CHECK-FIXES: Klazz<const MyStruct<0 < 1>> p39() {}

const Klazz<const MyStruct<(0 > 1)>> p40() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const MyStru
// CHECK-FIXES: Klazz<const MyStruct<(0 > 1)>> p40() {}

// re-declaration of p15.
const int p15();
// CHECK-FIXES: int p15();

const int p15() {
// CHECK-MESSAGES: [[@LINE-1]]:1: warning:
// CHECK-FIXES: int p15() {
  return 0;
}

// Exercise the lexer.

const /* comment */ /* another comment*/ int p16() { return 0; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning:
// CHECK-FIXES: /* comment */ /* another comment*/ int p16() { return 0; }

/* comment */ const
// CHECK-MESSAGES: [[@LINE-1]]:15: warning:
// CHECK-FIXES: /* comment */
// more
/* another comment*/ int p17() { return 0; }

// Test cases where the `const` token lexically is hidden behind some form of
// indirection.

#define CONSTINT const int
CONSTINT p18() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu

#define CONST const
CONST int p19() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu

using ty = const int;
ty p21() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'ty' (aka 'const int') is

typedef const int ty2;
ty2 p22() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'ty2' (aka 'const int') i

// Declaration uses a macro, while definition doesn't.  In this case, we won't
// fix the declaration, and will instead issue a warning.
CONST int p23();
// CHECK-NOTE: [[@LINE-1]]:1: note: could not transform this declaration

const int p23();
// CHECK-FIXES: int p23();

const int p23() { return 3; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
// CHECK-FIXES: int p23() { return 3; }

int const p24() { return 3; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
// CHECK-FIXES: int p24() { return 3; }

int const * const p25(const int* p) { return p; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int *const' is 'co
// CHECK-FIXES: int const * p25(const int* p) { return p; }

// We cannot (yet) fix instances that use trailing return types, but we can
// warn.
auto p26() -> const int { return 3; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
auto p27() -> int const { return 3; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu

std::add_const<int>::type p28() { return 3; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'std::add_const<int>::typ

// p29, p30 are based on
// llvm/projects/test-suite/SingleSource/Benchmarks/Misc-C++-EH/spirit.cpp:
template <class T>
Klazz<T const> const p29(T const &t) { return {}; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const T>' is
// CHECK-FIXES: Klazz<T const> p29(T const &t) { return {}; }

Klazz<char const *> const p30(char const *s) { return s; }
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const Klazz<const char *
// CHECK-FIXES: Klazz<char const *> p30(char const *s) { return s; }

const int n1 = 1;
const Clazz n2 = Clazz();
const Clazz* n3 = new Clazz();
Clazz *const n4 = new Clazz();
const Clazz *const n5 = new Clazz();
constexpr int n6 = 6;
constexpr int n7() { return 8; }
const int eight = 8;
constexpr const int* n8() { return &eight; }
Klazz<const int> n9();
const Klazz<const int>* n10();
const Klazz<const int>& Clazz::n11(const Klazz<const int>) const {}

// Declaration only.
const int n14();

int **const * n_multiple_ptr();
int *const & n_pointer_ref();

class PVBase {
public:
  virtual const int getC() = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const int' is 'const'-qualified at the top level, which may reduce code readability without improving const correctness
  // CHECK-NOT-FIXES: virtual int getC() = 0;
};

class PVDerive : public PVBase {
public:
  const int getC() { return 1; }
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: return type 'const int' is 'const'-qualified at the top level, which may reduce code readability without improving const correctness
  // CHECK-NOT-FIXES: int getC() { return 1; }
};

// Don't warn about const auto types, because it may be impossible to make them non-const
// without a significant semantics change. Since `auto` drops cv-qualifiers,
// tests check `decltype(auto)`.
decltype(auto) n16() {
  static const int i = 42;
  return i;
}

// Don't warn about `decltype(<expr>)` types
const int n17i = 1;
decltype(n17i) n17() {
  return 17;
}

// Do warn when on decltype types with the local const qualifier
// `const decltype(auto)` won't compile, so check only `const decltype(<expr>)`
const decltype(n17i) n18() {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const decltype(n17i)
  // CHECK-FIXES: decltype(n17i) n18() {
  return 18;
}

// `volatile` modifier doesn't affect the checker
volatile decltype(n17i) n19() {
  return 19;
}

// Don't warn about `__typeof__(<expr>)` types
__typeof__(n17i) n20() {
  return 20;
}

// Don't warn about `__typeof__(type)` types
__typeof__(const int) n21() {
  return 21;
}
