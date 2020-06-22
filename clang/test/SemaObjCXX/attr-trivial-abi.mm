// RUN: %clang_cc1 -std=c++11 -fobjc-runtime-has-weak -fobjc-weak -fobjc-arc -fsyntax-only -verify %s

void __attribute__((trivial_abi)) foo(); // expected-warning {{'trivial_abi' attribute only applies to classes}}

struct [[clang::trivial_abi]] S0 {
  int a;
};

struct __attribute__((trivial_abi)) S1 {
  int a;
};

struct __attribute__((trivial_abi)) S2 { // expected-warning {{'trivial_abi' cannot be applied to 'S2'}} expected-note {{has a __weak field}}
  __weak id a;
};

struct __attribute__((trivial_abi)) S3 { // expected-warning {{'trivial_abi' cannot be applied to 'S3'}} expected-note {{is polymorphic}}
  virtual void m();
};

struct S3_2 {
  virtual void m();
} __attribute__((trivial_abi)); // expected-warning {{'trivial_abi' cannot be applied to 'S3_2'}} expected-note {{is polymorphic}}

struct S4 {
  int a;
};

struct __attribute__((trivial_abi)) S5 : public virtual S4 { // expected-warning {{'trivial_abi' cannot be applied to 'S5'}} expected-note {{has a virtual base}}
};

struct __attribute__((trivial_abi)) S9 : public S4 {
};

struct S6 {
  __weak id a;
};

struct __attribute__((trivial_abi)) S12 { // expected-warning {{'trivial_abi' cannot be applied to 'S12'}} expected-note {{has a __weak field}}
  __weak id a;
};

struct __attribute__((trivial_abi)) S13 { // expected-warning {{'trivial_abi' cannot be applied to 'S13'}} expected-note {{has a __weak field}}
  __weak id a[2];
};

struct __attribute__((trivial_abi)) S7 { // expected-warning {{'trivial_abi' cannot be applied to 'S7'}} expected-note {{has a field of a non-trivial class type}}
  S6 a;
};

struct __attribute__((trivial_abi)) S11 { // expected-warning {{'trivial_abi' cannot be applied to 'S11'}} expected-note {{has a field of a non-trivial class type}}
  S6 a[2];
};

struct __attribute__((trivial_abi(1))) S8 { // expected-error {{'trivial_abi' attribute takes no arguments}}
  int a;
};

// Do not warn when 'trivial_abi' is used to annotate a template class.
template<class T>
struct __attribute__((trivial_abi)) S10 {
  T p;
};

S10<int *> p1;
S10<__weak id> p2;

template<>
struct __attribute__((trivial_abi)) S10<id> { // expected-warning {{'trivial_abi' cannot be applied to 'S10<id>'}} expected-note {{has a __weak field}}
  __weak id a;
};

template<class T>
struct S14 {
  T a;
  __weak id b;
};

template<class T>
struct __attribute__((trivial_abi)) S15 : S14<T> {
};

S15<int> s15;

template<class T>
struct __attribute__((trivial_abi)) S16 {
  S14<T> a;
};

S16<int> s16;

template<class T>
struct __attribute__((trivial_abi)) S17 { // expected-warning {{'trivial_abi' cannot be applied to 'S17'}} expected-note {{has a __weak field}}
  __weak id a;
};

S17<int> s17;

namespace deletedCopyMoveConstructor {
  struct __attribute__((trivial_abi)) CopyMoveDeleted { // expected-warning {{'trivial_abi' cannot be applied to 'CopyMoveDeleted'}} expected-note {{copy constructors and move constructors are all deleted}}
    CopyMoveDeleted(const CopyMoveDeleted &) = delete;
    CopyMoveDeleted(CopyMoveDeleted &&) = delete;
  };

  struct __attribute__((trivial_abi)) S18 { // expected-warning {{'trivial_abi' cannot be applied to 'S18'}} expected-note {{copy constructors and move constructors are all deleted}}
    CopyMoveDeleted a;
  };

  struct __attribute__((trivial_abi)) CopyDeleted {
    CopyDeleted(const CopyDeleted &) = delete;
    CopyDeleted(CopyDeleted &&) = default;
  };

  struct __attribute__((trivial_abi)) MoveDeleted {
    MoveDeleted(const MoveDeleted &) = default;
    MoveDeleted(MoveDeleted &&) = delete;
  };

  struct __attribute__((trivial_abi)) S19 { // expected-warning {{'trivial_abi' cannot be applied to 'S19'}} expected-note {{copy constructors and move constructors are all deleted}}
    CopyDeleted a;
    MoveDeleted b;
  };

  // This is fine since the move constructor isn't deleted.
  struct __attribute__((trivial_abi)) S20 {
    int &&a; // a member of rvalue reference type deletes the copy constructor.
  };
}
