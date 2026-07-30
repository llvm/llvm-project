// RUN: %clang_cc1 -std=c++2a -verify %s

namespace Rel {
  struct A {
    int n;
    constexpr int operator<=>(A a) const { return n - a.n; }
    friend bool operator<(const A&, const A&) = default;
    friend bool operator<=(const A&, const A&) = default;
    friend bool operator>(const A&, const A&) = default;
    friend bool operator>=(const A&, const A&) = default;
  };
  static_assert(A{0} < A{1});
  static_assert(A{1} < A{1}); // expected-error {{failed}}
  static_assert(A{0} <= A{1});
  static_assert(A{1} <= A{1});
  static_assert(A{2} <= A{1}); // expected-error {{failed}}
  static_assert(A{1} > A{0});
  static_assert(A{1} > A{1}); // expected-error {{failed}}
  static_assert(A{1} >= A{0});
  static_assert(A{1} >= A{1});
  static_assert(A{1} >= A{2}); // expected-error {{failed}}

  struct B {
    bool operator<=>(B) const = delete; // expected-note 4{{deleted here}} expected-note-re 8{{candidate {{.*}} deleted}}
    friend bool operator<(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note {{because it would invoke a deleted comparison}} expected-note-re {{candidate {{.*}} deleted}} expected-note{{replace 'default'}}
    friend bool operator<=(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note {{because it would invoke a deleted comparison}} expected-note-re {{candidate {{.*}} deleted}} expected-note{{replace 'default'}}
    friend bool operator>(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note {{because it would invoke a deleted comparison}} expected-note-re {{candidate {{.*}} deleted}} expected-note{{replace 'default'}}
    friend bool operator>=(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note {{because it would invoke a deleted comparison}} expected-note-re {{candidate {{.*}} deleted}} expected-note{{replace 'default'}}
  };
  bool b1 = B() < B(); // expected-error {{deleted}}
  bool b2 = B() <= B(); // expected-error {{deleted}}
  bool b3 = B() > B(); // expected-error {{deleted}}
  bool b4 = B() >= B(); // expected-error {{deleted}}

  struct C {
    friend bool operator<=>(const C&, const C&);
    friend bool operator<(const C&, const C&); // expected-note {{because this non-rewritten comparison function would be the best match}}

    bool operator<(const C&) const = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
    bool operator>(const C&) const = default; // OK
  };
}

// Under P2002R0, operator!= follows these rules too.
namespace NotEqual {
  struct A {
    int n;
    constexpr bool operator==(A a) const { return n == a.n; }
    friend bool operator!=(const A&, const A&) = default;
  };
  static_assert(A{1} != A{2});
  static_assert(A{1} != A{1}); // expected-error {{failed}}

  struct B {
    bool operator==(B) const = delete; // expected-note {{deleted here}} expected-note-re 2{{candidate {{.*}} deleted}}
    friend bool operator!=(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note {{because it would invoke a deleted comparison}} expected-note-re {{candidate {{.*}} deleted}} expected-note{{replace 'default'}}
  };
  bool b = B() != B(); // expected-error {{deleted}}

  struct C {
    friend bool operator==(const C&, const C&);
    friend bool operator!=(const C&, const C&); // expected-note {{because this non-rewritten comparison function would be the best match}}

    bool operator!=(const C&) const = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  };

  // Ensure we don't go into an infinite loop diagnosing this: the first function
  // is deleted because it calls the second function, which is deleted because it
  // calls the first.
  struct Evil {
    friend bool operator!=(const Evil&, const Evil&) = default; // expected-warning {{implicitly deleted}} expected-note {{would be the best match}} expected-note{{replace 'default'}}
    bool operator!=(const Evil&) const = default; // expected-warning {{implicitly deleted}} expected-note {{would be the best match}} expected-note{{replace 'default'}}
  };
}

namespace Access {
  class A {
    int operator<=>(A) const; // expected-note {{private}}
  };
  struct B : A {
    friend bool operator<(const B&, const B&) = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
    // expected-note@-1 {{defaulted 'operator<' is implicitly deleted because it would invoke a private 'operator<=>' member of 'Access::A'}}
  };
}
