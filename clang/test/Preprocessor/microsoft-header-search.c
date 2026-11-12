// RUN: %clang_cc1 -I%S/Inputs/microsoft-header-search %s -fms-compatibility -verify

// expected-warning@Inputs/microsoft-header-search/a/findme.h:3 {{findme.h successfully included using Microsoft header search rules}}
// expected-warning@Inputs/microsoft-header-search/a/b/include3.h:3 {{#include resolved using non-portable Microsoft search rules as}}

// expected-warning@Inputs/microsoft-header-search/falsepos.h:3 {{successfully resolved the falsepos.h header}}

#include "Inputs/microsoft-header-search/include1.h"
