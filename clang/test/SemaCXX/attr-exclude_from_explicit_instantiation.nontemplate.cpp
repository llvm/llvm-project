// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that exclude_from_explicit_instantiation is warned if attached
// on a non-template context or on a non-member entity.

#define EXCLUDE_ATTR __attribute__((exclude_from_explicit_instantiation))

struct C {
  EXCLUDE_ATTR void fn_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  EXCLUDE_ATTR static int var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  struct EXCLUDE_ATTR nested_excluded { // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
    EXCLUDE_ATTR void fn_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
    EXCLUDE_ATTR static int var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  };
  struct nested {
    EXCLUDE_ATTR void fn_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
    EXCLUDE_ATTR static int var_excluded;  // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  };
  template <class T>
  struct EXCLUDE_ATTR class_template_excluded {}; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  template <class T>
  EXCLUDE_ATTR static T var_template_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
  template <class T>
  EXCLUDE_ATTR void fn_template_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-template context}}
};

struct EXCLUDE_ATTR class_excluded {}; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
EXCLUDE_ATTR int var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
EXCLUDE_ATTR void fn_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}

template <class T>
struct EXCLUDE_ATTR class_template_excluded {}; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
template <class T>
EXCLUDE_ATTR T var_template_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
template <class T>
EXCLUDE_ATTR void fn_template_excluded(); // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}

void fn () {
  EXCLUDE_ATTR static int var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
}

auto lambda = [](auto x) {
  EXCLUDE_ATTR static int var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
};

template <class T>
void fn_template() {
  EXCLUDE_ATTR static T var_excluded; // expected-warning{{'exclude_from_explicit_instantiation' attribute ignored on a non-member declaration}}
};
