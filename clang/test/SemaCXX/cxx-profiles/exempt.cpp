// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles -fsyntax-only \
// RUN:   -isystem %S/Inputs/system -I %S/Inputs -verify %s

[[profiles::enforce(std::casting)]];
[[profiles::exempt(std::casting, quote_header: "cxx-profiles-exempt.h",
                   justification: "legacy interface")]];
[[profiles::exempt(std::casting)]];
// expected-error@-1 {{'profiles::exempt' attribute requires an 'angle_header' or 'quote_header' argument}}
[[profiles::exempt(std::casting, quote_header: unquoted)]];
// expected-error@-1 {{argument 'quote_header' of 'profiles::exempt' attribute must be a string literal}}

#include <cxx-profiles-system.h>
#include "cxx-profiles-exempt.h"
#include "cxx-profiles-not-exempt.h"

char *main_file_cast(int *p) {
  return reinterpret_cast<char *>(p);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}
