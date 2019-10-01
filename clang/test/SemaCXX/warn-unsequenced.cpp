// RUN: %clang_cc1 -fsyntax-only -verify=cxx11 -std=c++11 -Wno-unused -Wno-uninitialized \
// RUN:            -Wunsequenced -Wno-c++17-extensions -Wno-c++14-extensions %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx17 -std=c++17 -Wno-unused -Wno-uninitialized \
// RUN:            -Wunsequenced -Wno-c++17-extensions -Wno-c++14-extensions %s

int f(int, int = 0);

struct A {
  int x, y;
};
struct S {
  S(int, int);
  int n;
};

// TODO: Implement the C++17 sequencing rules.
void test() {
  int a;
  int xs[10];
  ++a = 0; // ok
  a + ++a; // cxx11-warning {{unsequenced modification and access to 'a'}}
           // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  a = ++a; // ok
  a + a++; // cxx11-warning {{unsequenced modification and access to 'a'}}
           // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  a = a++; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
           // TODO cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  ++ ++a; // ok
  (a++, a++); // ok
  ++a + ++a; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
             // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  a++ + a++; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
             // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  (a++, a) = 0; // ok, increment is sequenced before value computation of LHS
  a = xs[++a]; // ok
  a = xs[a++]; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
               // TODO cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  (a ? xs[0] : xs[1]) = ++a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                             // TODO cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  a = (++a, ++a); // ok
  a = (a++, ++a); // ok
  a = (a++, a++); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                  // TODO cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  f(a, a); // ok
  f(a = 0, a); // cxx11-warning {{unsequenced modification and access to 'a'}}
               // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  f(a, a += 0); // cxx11-warning {{unsequenced modification and access to 'a'}}
                // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  f(a = 0, a = 0); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                   // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  a = f(++a); // ok
  a = f(a++); // ok
  a = f(++a, a++); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                   // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}

  // Compound assignment "A OP= B" is equivalent to "A = A OP B" except that A
  // is evaluated only once.
  (++a, a) = 1; // ok
  (++a, a) += 1; // ok
  a = ++a; // ok
  a += ++a; // cxx11-warning {{unsequenced modification and access to 'a'}}
            // TODO cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  A agg1 = { a++, a++ }; // ok
  A agg2 = { a++ + a, a++ }; // cxx11-warning {{unsequenced modification and access to 'a'}}
                             // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  S str1(a++, a++); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                    // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  S str2 = { a++, a++ }; // ok
  S str3 = { a++ + a, a++ }; // cxx11-warning {{unsequenced modification and access to 'a'}}
                             // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  struct Z { A a; S s; } z = { { ++a, ++a }, { ++a, ++a } }; // ok
  a = S { ++a, a++ }.n; // ok
  A { ++a, a++ }.x; // ok
  a = A { ++a, a++ }.x; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                        // TODO cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  A { ++a, a++ }.x + A { ++a, a++ }.y; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                                       // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}

  (xs[2] && (a = 0)) + a; // ok
  (0 && (a = 0)) + a; // ok
  (1 && (a = 0)) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                      // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  (xs[3] || (a = 0)) + a; // ok
  (0 || (a = 0)) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                      // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  (1 || (a = 0)) + a; // ok

  (xs[4] ? a : ++a) + a; // ok
  (0 ? a : ++a) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                     // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  (1 ? a : ++a) + a; // ok
  (0 ? a : a++) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                     // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  (1 ? a : a++) + a; // ok
  (xs[5] ? ++a : ++a) + a; // FIXME: warn here

  (++a, xs[6] ? ++a : 0) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                              // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  // Here, the read of the fourth 'a' might happen before or after the write to
  // the second 'a'.
  a += (a++, a) + a; // cxx11-warning {{unsequenced modification and access to 'a'}}
                     // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}

  int *p = xs;
  a = *(a++, p); // ok
  a = a++ && a; // ok
  p[(long long unsigned)(p = 0)]; // cxx11-warning {{unsequenced modification and access to 'p'}}

  A *q = &agg1;
  (q = &agg2)->y = q->x; // cxx11-warning {{unsequenced modification and access to 'q'}}
                         // TODO cxx17-warning@-1 {{unsequenced modification and access to 'q'}}

  // This has undefined behavior if a == 0; otherwise, the side-effect of the
  // increment is sequenced before the value computation of 'f(a, a)', which is
  // sequenced before the value computation of the '&&', which is sequenced
  // before the assignment. We treat the sequencing in '&&' as being
  // unconditional.
  a = a++ && f(a, a);

  // This has undefined behavior if a != 0. FIXME: We should diagnose this.
  (a && a++) + a;

  (xs[7] && ++a) * (!xs[7] && ++a); // ok

  xs[0] = (a = 1, a); // ok
  (a -= 128) &= 128; // ok
  ++a += 1; // ok

  xs[8] ? ++a + a++ : 0; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                         // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  xs[8] ? 0 : ++a + a++; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                         // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  xs[8] ? ++a : a++; // ok

  xs[8] && (++a + a++); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                        // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  xs[8] || (++a + a++); // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                        // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}

  (__builtin_classify_type(++a) ? 1 : 0) + ++a; // ok
  (__builtin_constant_p(++a) ? 1 : 0) + ++a; // ok
  (__builtin_object_size(&(++a, a), 0) ? 1 : 0) + ++a; // ok
  (__builtin_expect(++a, 0) ? 1 : 0) + ++a; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
                                            // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
}

namespace members {

struct S1 {
  unsigned bf1 : 2;
  unsigned bf2 : 2;
  unsigned a;
  unsigned b;
  static unsigned x;
  void member_f(S1 &s);
};

void S1::member_f(S1 &s) {
  ++a + ++a; // cxx11-warning {{multiple unsequenced modifications to 'a'}}
             // cxx17-warning@-1 {{multiple unsequenced modifications to 'a'}}
  a + ++a; // cxx11-warning {{unsequenced modification and access to 'a'}}
           // cxx17-warning@-1 {{unsequenced modification and access to 'a'}}
  ++a + ++b; // no-warning
  a + ++b; // no-warning

  // TODO: Warn here.
  ++s.a + ++s.a; // no-warning TODO {{multiple unsequenced modifications to}}
  s.a + ++s.a; // no-warning TODO {{unsequenced modification and access to}}
  ++s.a + ++s.b; // no-warning
  s.a + ++s.b; // no-warning

  ++a + ++s.a; // no-warning
  a + ++s.a; // no-warning
  ++a + ++s.b; // no-warning
  a + ++s.b; // no-warning

  // TODO Warn here for bit-fields in the same memory location.
  ++bf1 + ++bf1; // cxx11-warning {{multiple unsequenced modifications to 'bf1'}}
                 // cxx17-warning@-1 {{multiple unsequenced modifications to 'bf1'}}
  bf1 + ++bf1; // cxx11-warning {{unsequenced modification and access to 'bf1'}}
               // cxx17-warning@-1 {{unsequenced modification and access to 'bf1'}}
  ++bf1 + ++bf2; // no-warning TODO {{multiple unsequenced modifications to}}
  bf1 + ++bf2; // no-warning TODO {{unsequenced modification and access to}}

  // TODO Warn here for bit-fields in the same memory location.
  ++s.bf1 + ++s.bf1; // no-warning TODO {{multiple unsequenced modifications to}}
  s.bf1 + ++s.bf1; // no-warning TODO {{unsequenced modification and access to}}
  ++s.bf1 + ++s.bf2; // no-warning TODO {{multiple unsequenced modifications to}}
  s.bf1 + ++s.bf2; // no-warning TODO {{unsequenced modification and access to}}

  ++bf1 + ++s.bf1; // no-warning
  bf1 + ++s.bf1; // no-warning
  ++bf1 + ++s.bf2; // no-warning
  bf1 + ++s.bf2; // no-warning

  struct Der : S1 {};
  Der d;
  Der &d_ref = d;
  S1 &s1_ref = d_ref;

  ++s1_ref.a + ++d_ref.a; // no-warning TODO {{multiple unsequenced modifications to member 'a' of 'd'}}
  ++s1_ref.a + d_ref.a; // no-warning TODO {{unsequenced modification and access to member 'a' of 'd'}}
  ++s1_ref.a + ++d_ref.b; // no-warning
  ++s1_ref.a + d_ref.b; // no-warning

  ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
             // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
  ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
           // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
  ++s.x + x; // no-warning TODO {{unsequenced modification and access to static member 'x' of 'S1'}}
  ++this->x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
                 // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
  ++d_ref.x + ++S1::x; // no-warning TODO {{unsequenced modification and access to static member 'x' of 'S1'}}
}

struct S2 {
  union { unsigned x, y; };
  void f2();
};

void S2::f2() {
  ++x + ++x; // no-warning TODO {{multiple unsequenced modifications to}}
  x + ++x; // no-warning TODO {{unsequenced modification and access to}}
  ++x + ++y; // no-warning
  x + ++y; // no-warning
}

void f2(S2 &s) {
  ++s.x + ++s.x; // no-warning TODO {{multiple unsequenced modifications to}}
  s.x + ++s.x; // no-warning TODO {{unsequenced modification and access to}}
  ++s.x + ++s.y; // no-warning
  s.x + ++s.y; // no-warning
}

struct S3 {
  union {
    union {
      unsigned x;
    };
  };
  unsigned y;
  void f3();
};

void S3::f3() {
  ++x + ++x; // no-warning TODO {{multiple unsequenced modifications to}}
  x + ++x; // no-warning TODO {{unsequenced modification and access to}}
  ++x + ++y; // no-warning
  x + ++y; // no-warning
}

void f3(S3 &s) {
  ++s.x + ++s.x; // no-warning TODO {{multiple unsequenced modifications to}}
  s.x + ++s.x; // no-warning TODO {{unsequenced modification and access to}}
  ++s.x + ++s.y; // no-warning
  s.x + ++s.y; // no-warning
}

struct S4 : S3 {
  unsigned y;
  void f4();
};

void S4::f4() {
  ++x + ++x; // no-warning TODO {{multiple unsequenced modifications to}}
  x + ++x; // no-warning TODO {{unsequenced modification and access to}}
  ++x + ++y; // no-warning
  x + ++y; // no-warning
  ++S3::y + ++y; // no-warning
  S3::y + ++y; // no-warning
}

void f4(S4 &s) {
  ++s.x + ++s.x; // no-warning TODO {{multiple unsequenced modifications to}}
  s.x + ++s.x; // no-warning TODO {{unsequenced modification and access to}}
  ++s.x + ++s.y; // no-warning
  s.x + ++s.y; // no-warning
  ++s.S3::y + ++s.y; // no-warning
  s.S3::y + ++s.y; // no-warning
}

static union {
  unsigned Ux;
  unsigned Uy;
};

void f5() {
  ++Ux + ++Ux; // no-warning TODO {{multiple unsequenced modifications to}}
  Ux + ++Ux; // no-warning TODO {{unsequenced modification and access to}}
  ++Ux + ++Uy; // no-warning
  Ux + ++Uy; // no-warning
}

void f6() {
  struct S { unsigned x, y; } s;
  ++s.x + ++s.x; // no-warning TODO {{multiple unsequenced modifications to}}
  s.x + ++s.x; // no-warning TODO {{unsequenced modification and access to}}
  ++s.x + ++s.y; // no-warning
  s.x + ++s.y; // no-warning

  struct { unsigned x, y; } t;
  ++t.x + ++t.x; // no-warning TODO {{multiple unsequenced modifications to}}
  t.x + ++t.x; // no-warning TODO {{unsequenced modification and access to}}
  ++t.x + ++t.y; // no-warning
  t.x + ++t.y; // no-warning
}

} // namespace members

namespace references {
void reference_f() {
  // TODO: Check that we can see through references.
  // For now this is completely unhandled.
  int a;
  int xs[10];
  int &b = a;
  int &c = b;
  int &ra1 = c;
  int &ra2 = b;
  int other;

  ++ra1 + ++ra2; // no-warning TODO {{multiple unsequenced modifications to}}
  ra1 + ++ra2; // no-warning TODO {{unsequenced modification and access to}}
  ++ra1 + ++other; // no-warning
  ra1 + ++other; // no-warning

  // Make sure we handle reference cycles.
  int &ref_cycle = ref_cycle;
  ++ref_cycle + ++ref_cycle; // cxx11-warning {{multiple unsequenced modifications to 'ref_cycle'}}
                             // cxx17-warning@-1 {{multiple unsequenced modifications to 'ref_cycle'}}
  ref_cycle + ++ref_cycle; // cxx11-warning {{unsequenced modification and access to 'ref_cycle'}}
                           // cxx17-warning@-1 {{unsequenced modification and access to 'ref_cycle'}}
}
} // namespace references

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename> struct tuple_size;
  template<size_t, typename> struct tuple_element { using type = int; };
}
namespace bindings {

  struct A { int x, y; };
  typedef int B[2];
  struct C { template<int> int get(); };
  struct D : A {};

} // namespace bindings
template<> struct std::tuple_size<bindings::C> { enum { value = 2 }; };
namespace bindings {
void testa() {
  A a;
  {
    auto [x, y] = a;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++a.x; // no-warning
    ++x + a.x; // no-warning
  }
  {
    auto &[x, y] = a;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++a.x; // no-warning TODO
    ++x + a.x; // no-warning TODO
  }
}
void testb() {
  B b;
  {
    auto [x, y] = b;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++b[0]; // no-warning
    ++x + b[0]; // no-warning
  }
  {
    auto &[x, y] = b;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++b[0]; // no-warning TODO
    ++x + b[0]; // no-warning TODO
  }
}
void testc() {
  C c;
  {
    auto [x, y] = c;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
  }
  {
    auto &[x, y] = c;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
  }
}
void testd() {
  D d;
  {
    auto [x, y] = d;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++d.x; // no-warning
    ++x + d.x; // no-warning
  }
  {
    auto &[x, y] = d;
    ++x + ++x; // cxx11-warning {{multiple unsequenced modifications to 'x'}}
               // cxx17-warning@-1 {{multiple unsequenced modifications to 'x'}}
    ++x + x; // cxx11-warning {{unsequenced modification and access to 'x'}}
             // cxx17-warning@-1 {{unsequenced modification and access to 'x'}}
    ++x + ++y; // no-warning
    ++x + y; // no-warning
    ++x + ++d.x; // no-warning TODO
    ++x + d.x; // no-warning TODO
  }
}
} // namespace bindings

namespace templates {

template <typename T>
struct Bar {
  T get() { return 0; }
};

template <typename X>
struct Foo {
  int Run();
  Bar<int> bar;
};

enum E {e1, e2};
bool operator&&(E, E);

void foo(int, int);

template <typename X>
int Foo<X>::Run() {
  char num = 0;

  // Before instantiation, Clang may consider the builtin operator here as
  // unresolved function calls, and treat the arguments as unordered when
  // the builtin operator evaluatation is well-ordered.  Waiting until
  // instantiation to check these expressions will prevent false positives.
  if ((num = bar.get()) < 5 && num < 10) { }
  if ((num = bar.get()) < 5 || num < 10) { }
  if (static_cast<E>((num = bar.get()) < 5) || static_cast<E>(num < 10)) { }

  if (static_cast<E>((num = bar.get()) < 5) && static_cast<E>(num < 10)) { }
  // cxx11-warning@-1 {{unsequenced modification and access to 'num'}}
  // cxx17-warning@-2 {{unsequenced modification and access to 'num'}}

  foo(num++, num++);
  // cxx11-warning@-1 {{multiple unsequenced modifications to 'num'}}
  // cxx17-warning@-2 {{multiple unsequenced modifications to 'num'}}
  return 1;
}

int x = Foo<int>().Run();
// cxx11-note@-1 {{in instantiation of member function 'templates::Foo<int>::Run'}}
// cxx17-note@-2 {{in instantiation of member function 'templates::Foo<int>::Run'}}


template <typename T>
int Run2() {
  T t = static_cast<T>(0);
  return (t = static_cast<T>(1)) && t;
  // cxx11-warning@-1 {{unsequenced modification and access to 't'}}
  // cxx17-warning@-2 {{unsequenced modification and access to 't'}}
}

int y = Run2<bool>();
int z = Run2<E>();
// cxx11-note@-1{{in instantiation of function template specialization 'templates::Run2<templates::E>' requested here}}
// cxx17-note@-2{{in instantiation of function template specialization 'templates::Run2<templates::E>' requested here}}

template <typename T> int var = sizeof(T);
void test_var() {
  var<int>++ + var<int>++; // cxx11-warning {{multiple unsequenced modifications to 'var<int>'}}
                           // cxx17-warning@-1 {{multiple unsequenced modifications to 'var<int>'}}
  var<int>++ + var<int>; // cxx11-warning {{unsequenced modification and access to 'var<int>'}}
                         // cxx17-warning@-1 {{unsequenced modification and access to 'var<int>'}}
  int &r = var<int>;
  r++ + var<int>++; // no-warning TODO {{multiple unsequenced modifications to 'var<int>'}}
  r++ + var<long>++; // no-warning
}

} // namespace templates
