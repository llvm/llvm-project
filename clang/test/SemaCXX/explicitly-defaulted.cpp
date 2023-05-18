// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct Deleted {
  Deleted() = delete; // expected-note 3{{marked deleted here}}
  Deleted(const Deleted &) = delete; // expected-note 2{{marked deleted here}}
  Deleted(Deleted &&) = delete; // expected-note 2{{marked deleted here}}
  Deleted &operator=(const Deleted &) = delete; // expected-note 2{{marked deleted here}}
  Deleted &operator=(Deleted &&) = delete; // expected-note 2{{marked deleted here}}
  ~Deleted() = delete; // expected-note 2{{marked deleted here}}
};

struct Derive : Deleted { // expected-note 6{{because base class}}
  Derive() = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Derive(const Derive &) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Derive(Derive &&) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Derive &operator=(const Derive &) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Derive &operator=(Derive &&) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  ~Derive() = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
};

struct Member {
  Deleted A; // expected-note 6{{because field 'A'}}
  Member() = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Member(const Member &) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Member(Member &&) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Member &operator=(const Member &) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  Member &operator=(Member &&) = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
  ~Member() = default; // expected-warning{{explicitly defaulted}} expected-note{{replace 'default' with 'delete'}}
};

template<typename T>
struct TDerive : T { // expected-note {{because base class}}
  TDerive() = default; //expected-note {{explicitly defaulted}} // Don't expect a fix note to be emitted
};

using ShouldDelete = TDerive<Deleted>;

ShouldDelete A; // expected-error{{call to implicitly-deleted}}
