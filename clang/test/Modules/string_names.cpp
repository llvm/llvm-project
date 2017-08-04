// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fmodules-decluse -I %S/Inputs/string_names %s -fmodule-name="my/module-a" -verify

// Check that we can preprocess with string module names.
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/string_names %s -fmodule-name="my/module-a" -E -frewrite-imports -o %t/test.ii
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-decluse -I %S/Inputs/string_names %t/test.ii -fmodule-name="my/module-a"

#include "a.h"
#include "b.h" // expected-error {{does not depend on a module exporting}}
#include "c.h"
