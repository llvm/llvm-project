// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -I%S/Inputs/header-attribs -fmodule-map-file=%S/Inputs/header-attribs/textual.modulemap -fmodules-cache-path=%t -verify %s -fmodule-name=A -fmodules-strict-decluse
// RUN: not %clang_cc1 -fmodules -I%S/Inputs/header-attribs -emit-module -x c++-module-map %S/Inputs/header-attribs/modular.modulemap -fmodules-cache-path=%t -fmodule-name=A 2>&1 | FileCheck %s --check-prefix BUILD-MODULAR

#include "foo.h" // ok, stats match
#include "bar.h" // expected-error {{does not depend on a module exporting 'bar.h'}}
#include "baz.h" // expected-error {{does not depend on a module exporting 'baz.h'}}

// FIXME: Explain why the 'bar.h' found on disk doesn't match the module map.
// BUILD-MODULAR: error: header 'bar.h' not found
