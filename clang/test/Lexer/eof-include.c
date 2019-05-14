// RUN: %clang_cc1 %s -verify
// vim: set binary noeol:

// This file intentionally ends without a \n on the last line.  Make sure your
// editor doesn't add one.

// expected-error@+1{{expected '>'}} expected-note@+1{{to match this '<'}}
#include <\
