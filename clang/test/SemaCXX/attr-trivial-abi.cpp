// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void __attribute__((trivial_abi)) foo(); // expected-warning {{'trivial_abi' attribute only applies to classes}}

// Should not crash.
template <class>
class __attribute__((trivial_abi)) a { a(a &&); };

struct [[clang::trivial_abi]] S0 {
  int a;
};

struct __attribute__((trivial_abi)) S1 {
  int a;
};

struct __attribute__((trivial_abi)) S3 { // expected-warning {{'trivial_abi' cannot be applied to 'S3'}} expected-note {{is polymorphic}}
  virtual void m();
};

struct S3_2 {
  virtual void m();
} __attribute__((trivial_abi)); // expected-warning {{'trivial_abi' cannot be applied to 'S3_2'}} expected-note {{is polymorphic}}

struct __attribute__((trivial_abi)) S3_3 { // expected-warning {{'trivial_abi' cannot be applied to 'S3_3'}} expected-note {{has a field of a non-trivial class type}}
  S3_3(S3_3 &&);
  S3_2 s32;
};

// Diagnose invalid trivial_abi even when the type is templated because it has a non-trivial field.
template <class T>
struct __attribute__((trivial_abi)) S3_4 { // expected-warning {{'trivial_abi' cannot be applied to 'S3_4'}} expected-note {{has a field of a non-trivial class type}}
  S3_4(S3_4 &&);
  S3_2 s32;
};

struct S4 {
  int a;
};

struct __attribute__((trivial_abi)) S5 : public virtual S4 { // expected-warning {{'trivial_abi' cannot be applied to 'S5'}} expected-note {{has a virtual base}}
};

struct __attribute__((trivial_abi)) S9 : public S4 {
};

struct __attribute__((trivial_abi(1))) S8 { // expected-error {{'trivial_abi' attribute takes no arguments}}
  int a;
};

// Do not warn about deleted ctors  when 'trivial_abi' is used to annotate a template class.
template <class T>
struct __attribute__((trivial_abi)) S10 {
  T p;
};

S10<int *> p1;

template <class T>
struct S14 {
  T a;
};

template <class T>
struct __attribute__((trivial_abi)) S15 : S14<T> {
};

S15<int> s15;

template <class T>
struct __attribute__((trivial_abi)) S16 {
  S14<T> a;
};

S16<int> s16;

template <class T>
struct __attribute__((trivial_abi)) S17 {
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
} // namespace deletedCopyMoveConstructor
