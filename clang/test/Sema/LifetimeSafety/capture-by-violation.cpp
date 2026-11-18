// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-capture-by-violation -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-annotation-verification -verify=expected,verification %s

#include "Inputs/lifetime-analysis.h"

struct Cap { const int *p; };
const int *global; // expected-note 4 {{escapes to this global storage}}
void takeString(std::string &&);
const int &id(const int &v [[clang::lifetimebound]]);

struct S {
  const int *field;                    // expected-note 6 {{escapes to this field}} verification-note {{escapes to this field}}
  std::string_view view;               // expected-note {{escapes to this field}}
  std::vector<std::string_view> views; // expected-note {{escapes to this field}}
  static const int *static_field;      // expected-note {{escapes to this static storage}}

  void captured(const int &x [[clang::lifetime_capture_by_this]]);

  void field_store(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    field = &x;
  }

  void also_into_param(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    c.p = &x;
    field = &x;
  }

  void into_param(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) {
    c.p = &x;
  }

  void conditional(bool b, Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    if (b)
      field = &x;
  }

  void through_lifetimebound_call(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    field = &id(x);
  }

  // A capture by this object in a call has no field to point at.
  void call_capturing_this(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    captured(x);
  }

  // `push_back` of a standard container is inferred to capture by `this`.
  void into_container(Cap &c, std::string_view s [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'s' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    views.push_back(s);
  }

  void captured_by_this(const int &x [[clang::lifetime_capture_by_this]]) {
    field = &x;
  }

  void captured_by_param_and_this(
      Cap &c,
      const int &x [[clang::lifetime_capture_by(c), clang::lifetime_capture_by_this]]) {
    field = &x;
  }

  void deprecated_this(const int &x [[clang::lifetime_capture_by(this)]]) { // expected-warning {{'lifetime_capture_by(this)' is deprecated; use 'lifetime_capture_by_this' instead}}
    field = &x;
  }

  void captured_by_unknown(const int &x [[clang::lifetime_capture_by_unknown]]) {
    field = &x;
    global = &x;
  }

  void into_global(const int &x [[clang::lifetime_capture_by_this]]) { // expected-warning {{'x' is captured by global storage but is not marked [[clang::lifetime_capture_by_global]]}}
    global = &x;
  }

  void into_static_member(const int &x [[clang::lifetime_capture_by_this]]) { // expected-warning {{'x' is captured by global storage but is not marked [[clang::lifetime_capture_by_global]]}}
    static_field = &x;
  }

  void into_this_and_global(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}} \
                                                                                     // expected-warning {{'x' is captured by global storage but is not marked [[clang::lifetime_capture_by_global]]}}
    field = &x;
    global = &x;
  }

  static void static_into_global(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by global storage but is not marked [[clang::lifetime_capture_by_global]]}}
    global = &x;
  }

  // An unnamed parameter cannot be referred to, so it is not captured.
  void unnamed(Cap &c, const int & [[clang::lifetime_capture_by(c)]]) {
    field = nullptr;
  }

  void view_store(Cap &c, std::string &s [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'s' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    view = s;
  }

  // The ownership of a moved argument is transferred.
  void moved(Cap &c, std::string &s [[clang::lifetime_capture_by(c)]]) {
    view = s;
    takeString(std::move(s));
  }

  void noescape(Cap &c, const int &x [[clang::noescape, clang::lifetime_capture_by(c)]]) { // verification-warning {{parameter is marked [[clang::noescape]] but escapes}}
    field = &x;
  }

  void unannotated(const int &x) {
    field = &x;
  }

  void declared_only(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]); // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}

  // FIXME: The capturers of a lambda's parameters are never resolved, so they
  // are not verified.
  void lambda() {
    auto L = [](Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) {
      global = &x;
    };
  }
};

void S::declared_only(Cap &c, const int &x) {
  field = &x;
}

struct T {
  const int *field; // expected-note {{escapes to this field}}
  T(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    field = &x;
  }
  T(const int &x [[clang::lifetime_capture_by_this]]) : field(&x) {}
  T(Cap &c, const int &x [[clang::lifetime_capture_by(c), clang::lifetimebound]], int)
      : field(&x) {}
};

template <typename U> struct TS {
  const U *field; // expected-note {{escapes to this field}}
  void store(Cap &c, const U &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by this object but is not marked [[clang::lifetime_capture_by_this]]}}
    field = &x;
  }
};
template struct TS<int>; // expected-note {{in instantiation of member function 'TS<int>::store' requested here}}

struct [[gsl::Pointer(int)]] View {
  const int *p;
  // FIXME: Whole-object assignment through `*this` is not modeled, so the
  // capture by this object is not verified.
  void assign(Cap &c, const View &other [[clang::lifetime_capture_by(c)]]) { *this = other; }
  void assign_this(const View &other [[clang::lifetime_capture_by_this]]) { *this = other; }
};

void free_into_global(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{'x' is captured by global storage but is not marked [[clang::lifetime_capture_by_global]]}}
  global = &x;
}

void free_captured_by_global(const int &x [[clang::lifetime_capture_by_global]]) {
  global = &x;
}

void deprecated_global(const int &x [[clang::lifetime_capture_by(global)]]) { // expected-warning {{'lifetime_capture_by(global)' is deprecated; use 'lifetime_capture_by_global' instead}}
  global = &x;
}
