// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-capture-by-violation -verify %s

struct Cap { const int *p; };

struct S {
  const int *field;

  // The annotation names parameter 'c', but the body captures the borrow into
  // 'this' (a store into a field) -- the annotation does not name 'this'.
  void lies(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{the borrow from 'x' is captured into this object, but '[[clang::lifetime_capture_by]]' does not name 'this'}}
    field = &x;
  }

  // Also captured into 'c', but still stored into a field of 'this' while the
  // annotation omits 'this' -- flagged (the fix is to add 'this' to the list).
  void incomplete(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{the borrow from 'x' is captured into this object, but '[[clang::lifetime_capture_by]]' does not name 'this'}}
    c.p = &x;
    field = &x;
  }

  // Truthful: captured only into the named parameter 'c', no field-of-'this'
  // store, nothing flagged.
  void truthful(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) {
    c.p = &x;
  }

  // capture_by(this) matches a store into a field of 'this'.
  void captures_this(const int &x [[clang::lifetime_capture_by(this)]]) {
    field = &x;
  }

  // capture_by naming 'this' among other capturers still matches.
  void captures_this_and_param(Cap &c,
                               const int &x [[clang::lifetime_capture_by(this, c)]]) {
    field = &x;
  }

  // capture_by(unknown) already covers an unspecified capturer, including
  // 'this' -- not flagged.
  void captures_unknown(const int &x [[clang::lifetime_capture_by(unknown)]]) {
    field = &x;
  }

  // capture_by(global) is a separate concern -- not flagged here.
  void captures_global(const int &x [[clang::lifetime_capture_by(global)]]) {
    field = &x;
  }

  // A store into a field of another object (not 'this') is not a capture into
  // 'this' and is not flagged.
  void store_into_other(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) {
    c.p = &x;
  }
};

// A constructor that stores a capture_by(c) parameter into a field of 'this'.
struct T {
  const int *field;
  T(Cap &c, const int &x [[clang::lifetime_capture_by(c)]]) { // expected-warning {{the borrow from 'x' is captured into this object, but '[[clang::lifetime_capture_by]]' does not name 'this'}}
    field = &x;
  }
};
