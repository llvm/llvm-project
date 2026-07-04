// Tests the `requires` block form of module map declarations (as opposed to a
// member-level `requires`): when a block's feature list is unsatisfied, the
// wrapped modules are not created at all, so importing one yields an ordinary
// "module not found" rather than "module is unavailable".
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Compiled as Objective-C (no C++): the C++-guarded modules do not exist,
// while the !cplusplus-guarded module does.
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t %t/test.m -verify
//
// Compiled as Objective-C++20: the C++-guarded modules exist, while the
// !cplusplus-guarded module does not.
// RUN: %clang_cc1 -x objective-c++ -std=c++20 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t %t/test.m -verify

//--- module.modulemap
requires cplusplus {
  module CXXOnly { header "cxxonly.h" }
}

// Negation: only exists when C++ is *not* available.
requires !cplusplus {
  module NotCXX { header "notcxx.h" }
}

// Multi-feature list: every feature must hold.
requires cplusplus, cplusplus20 {
  module CXX20 { header "cxx20.h" }
}

// Nested blocks accumulate their features.
requires cplusplus {
  requires cplusplus20 {
    module Nested { header "nested.h" }
  }
}

// A block inside a module body gates a submodule; the enclosing module is
// unguarded and always exists.
module Outer {
  header "outer.h"
  requires cplusplus {
    module Inner { header "inner.h" }
  }
}

//--- cxxonly.h
//--- notcxx.h
//--- cxx20.h
//--- nested.h
//--- outer.h
//--- inner.h

//--- test.m
@import Outer; // OK in both dialects.

#ifdef __cplusplus
@import CXXOnly;     // OK
@import CXX20;       // OK
@import Nested;      // OK
@import Outer.Inner; // OK
// The negation block is inactive under C++, so its module was never created.
@import NotCXX; // expected-error {{module 'NotCXX' not found}}
#else
@import NotCXX; // OK
// The C++-guarded blocks are inactive, so their modules were never created.
// (err_module_not_found is fatal, so this must be the last import.)
@import CXXOnly; // expected-error {{module 'CXXOnly' not found}}
#endif
