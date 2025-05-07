// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Verify that we can include <lifetimebound.h>
#include <lifetimebound.h>

struct has_lifetimebound_method {
  const char* get_ptr(char* ptr __lifetimebound) const;
};

struct has_lifetime_capture_by_method {
  void take_ptr(char* ptr __lifetime_capture_by(this));
  void take_ptr(has_lifetimebound_method a, char* ptr __lifetime_capture_by(a));
};

struct has_noescape_method {
  void takes_noescape(char* ptr __noescape);
};
