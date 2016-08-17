// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
extern void f1(int *);
extern void f2(char *);

struct __attribute__((packed)) Arguable {
  int x;
  char c;
  static void foo();
};

extern void f3(void());

namespace Foo {
struct __attribute__((packed)) Arguable {
  char c;
  int x;
  static void foo();
};
}

struct Arguable *get_arguable();

void f4(int &);

void to_void(void *);

template <typename... T>
void sink(T...);

void g0() {
  {
    Foo::Arguable arguable;
    f1(&arguable.x);   // expected-warning {{packed member 'x' of class or structure 'Foo::Arguable'}}
    f2(&arguable.c);   // no-warning
    f3(&arguable.foo); // no-warning

    to_void(&arguable.x);                             // no-warning
    void *p1 = &arguable.x;                           // no-warning
    void *p2 = static_cast<void *>(&arguable.x);      // no-warning
    void *p3 = reinterpret_cast<void *>(&arguable.x); // no-warning
    void *p4 = (void *)&arguable.x;                   // no-warning
    sink(p1, p2, p3, p4);
  }
  {
    Arguable arguable1;
    Arguable &arguable(arguable1);
    f1(&arguable.x);   // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable.c);   // no-warning
    f3(&arguable.foo); // no-warning
  }
  {
    Arguable *arguable1;
    Arguable *&arguable(arguable1);
    f1(&arguable->x);   // expected-warning {{packed member 'x' of class or structure 'Arguable'}}
    f2(&arguable->c);   // no-warning
    f3(&arguable->foo); // no-warning
  }
}

struct __attribute__((packed)) A {
  int x;
  char c;

  int *f0() {
    return &this->x; // expected-warning {{packed member 'x' of class or structure 'A'}}
  }

  int *g0() {
    return &x; // expected-warning {{packed member 'x' of class or structure 'A'}}
  }

  char *h0() {
    return &c; // no-warning
  }
};

struct B : A {
  int *f1() {
    return &this->x; // expected-warning {{packed member 'x' of class or structure 'A'}}
  }

  int *g1() {
    return &x; // expected-warning {{packed member 'x' of class or structure 'A'}}
  }

  char *h1() {
    return &c; // no-warning
  }
};

template <typename Ty>
class __attribute__((packed)) S {
  Ty X;

public:
  const Ty *get() const {
    return &X; // expected-warning {{packed member 'X' of class or structure 'S<int>'}}
               // expected-warning@-1 {{packed member 'X' of class or structure 'S<float>'}}
  }
};

template <typename Ty>
void h(Ty *);

void g1() {
  S<int> s1;
  s1.get(); // expected-note {{in instantiation of member function 'S<int>::get'}}

  S<char> s2;
  s2.get();

  S<float> s3;
  s3.get(); // expected-note {{in instantiation of member function 'S<float>::get'}}
}
