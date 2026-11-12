// Test that pragma diagnostic mappings from an explicit module are not
// corrupted by the presence of non-affecting module map files.
//
// When non-affecting module map files are pruned, NonAffectingRanges becomes
// non-empty. Ensure that getAdjustedOffset is not incorrectly applied to
// file-internal byte offsets in WritePragmaDiagnosticMappings, corrupting the
// serialized diagnostic state transition offsets.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Build the module with a non-affecting module map present.
// RUN: %clang_cc1 -std=c++20 -fmodules \
// RUN:   -fmodule-map-file=%t/nonaffecting/module.modulemap \
// RUN:   -emit-module -fmodule-name=diag_pragma \
// RUN:   -x c++ %t/module.modulemap -o %t/diag_pragma.pcm

// Use the module and verify the warning is suppressed.
// RUN: %clang_cc1 -std=c++20 -fmodules \
// RUN:   -fmodule-file=%t/diag_pragma.pcm \
// RUN:   -I %t -verify %t/main.cpp

//--- module.modulemap
module diag_pragma { header "header.h" }

//--- header.h
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstring-plus-int"
template<typename T> const char *suppressed(T t) {
  return "foo" + t;
}
#pragma clang diagnostic pop

template<typename T> const char *unsuppressed(T t) {
  return "bar" + t;
}

//--- nonaffecting/module.modulemap
module nonaffecting {}

//--- main.cpp
#include "header.h"

void test() {
  suppressed(0);   // no warning expected - suppressed by pragma in module

  // expected-warning@header.h:9 {{adding 'int' to a string}}
  // expected-note@header.h:9 {{use array indexing}}
  unsuppressed(0); // expected-note {{in instantiation of}}
}
