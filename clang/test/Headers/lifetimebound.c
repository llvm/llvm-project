// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Verify that we can include <lifetimebound.h>
#include <lifetimebound.h>

struct foo {};

struct foo* get_foo(char *ptr __lifetimebound);

void non_escaping(char *ptr __noescape);
