// RUN: %clang_cc1 -fsyntax-only -verify -I %S/Inputs %s

#include <include_instead/bad-syntax.h>
#include <include_instead/non-system-header.h>

#include <include_instead/private1.h>
// expected-error@-1{{header '<include_instead/private1.h>' is an implementation detail; #include '<include_instead/public-before.h>' instead}}

#include "include_instead/private2.h"
// expected-error@-1{{header '"include_instead/private2.h"' is an implementation detail; #include either '<include_instead/public-before.h>' or '"include_instead/public-after.h"' instead}}

#include <include_instead/private3.h>
// expected-error@-1{{header '<include_instead/private3.h>' is an implementation detail; #include one of {'<include_instead/public-after.h>', '<include_instead/public-empty.h>', '"include_instead/public-before.h"'} instead}}

#include <include_instead/public-before.h>
#include <include_instead/public-after.h>
