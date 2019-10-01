// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct non_trivial {
  non_trivial();
  non_trivial(const non_trivial&);
  non_trivial& operator = (const non_trivial&);
  ~non_trivial();
};

union bad_union {
  non_trivial nt; // expected-note {{non-trivial default constructor}}
};
bad_union u; // expected-error {{call to implicitly-deleted default constructor}}
union bad_union2 { // expected-note {{all data members are const-qualified}}
  const int i;
};
bad_union2 u2; // expected-error {{call to implicitly-deleted default constructor}}

struct bad_anon {
  union {
    non_trivial nt; // expected-note {{non-trivial default constructor}}
  };
};
bad_anon a; // expected-error {{call to implicitly-deleted default constructor}}
struct bad_anon2 {
  union { // expected-note {{all data members of an anonymous union member are const-qualified}}
    const int i;
  };
};
bad_anon2 a2; // expected-error {{call to implicitly-deleted default constructor}}

// This would be great except that we implement
union good_union {
  const int i;
  float f;
};
good_union gu;
struct good_anon {
  union {
    const int i;
    float f;
  };
};
good_anon ga;

struct good : non_trivial {
  non_trivial nt;
};
good g;

struct bad_const {
  const good g; // expected-note {{field 'g' of const-qualified type 'const good' would not be initialized}}
};
bad_const bc; // expected-error {{call to implicitly-deleted default constructor}}

struct good_const {
  const non_trivial nt;
};
good_const gc;

struct no_default {
  no_default() = delete; // expected-note 5{{deleted here}}
};
struct no_dtor {
  ~no_dtor() = delete; // expected-note 2{{deleted here}}
};

struct bad_field_default {
  no_default nd; // expected-note {{field 'nd' has a deleted default constructor}}
};
bad_field_default bfd; // expected-error {{call to implicitly-deleted default constructor}}
struct bad_base_default : no_default { // expected-note {{base class 'no_default' has a deleted default constructor}}
};
bad_base_default bbd; // expected-error {{call to implicitly-deleted default constructor}}

struct bad_field_dtor {
  no_dtor nd; // expected-note {{field 'nd' has a deleted destructor}}
};
bad_field_dtor bfx; // expected-error {{call to implicitly-deleted default constructor}}
struct bad_base_dtor : no_dtor { // expected-note {{base class 'no_dtor' has a deleted destructor}}
};
bad_base_dtor bbx; // expected-error {{call to implicitly-deleted default constructor}}

struct ambiguous_default {
  ambiguous_default();
  ambiguous_default(int = 2);
};
struct has_amb_field {
  ambiguous_default ad; // expected-note {{field 'ad' has multiple default constructors}}
};
has_amb_field haf; // expected-error {{call to implicitly-deleted default constructor}}

class inaccessible_default {
  inaccessible_default();
};
struct has_inacc_field {
  inaccessible_default id; // expected-note {{field 'id' has an inaccessible default constructor}}
};
has_inacc_field hif; // expected-error {{call to implicitly-deleted default constructor}}

class friend_default {
  friend struct has_friend;
  friend_default();
};
struct has_friend {
  friend_default fd;
};
has_friend hf;

struct defaulted_delete {
  no_default nd; // expected-note 2{{because field 'nd' has a deleted default constructor}}
  defaulted_delete() = default; // expected-note{{implicitly deleted here}} expected-warning {{implicitly deleted}}
};
defaulted_delete dd; // expected-error {{call to implicitly-deleted default constructor}}

struct late_delete {
  no_default nd; // expected-note {{because field 'nd' has a deleted default constructor}}
  late_delete();
};
late_delete::late_delete() = default; // expected-error {{would delete it}}

// See also rdar://problem/8125400.
namespace empty {
  static union {}; // expected-warning {{does not declare anything}}
  static union { union {}; }; // expected-warning {{does not declare anything}}
  static union { struct {}; }; // expected-warning {{does not declare anything}}
  static union { union { union {}; }; }; // expected-warning {{does not declare anything}}
  static union { union { struct {}; }; }; // expected-warning {{does not declare anything}}
  static union { struct { union {}; }; }; // expected-warning {{does not declare anything}}
  static union { struct { struct {}; }; }; // expected-warning {{does not declare anything}}
}
