// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_enforced.cppm -o %t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_ok.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_fail.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_mismatch.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_repeated.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/impl_propagation.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_gmf_enforce.cppm -o %t/mod_gmf_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_gmf_ok.cpp -fmodule-file=GmfMod=%t/mod_gmf_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_gmf_only_enforce.cppm -o %t/mod_gmf_only_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_gmf_only_fail.cpp -fmodule-file=GmfOnlyMod=%t/mod_gmf_only_enforce.pcm -verify

// ===================================================================
// Module with enforced profiles
// ===================================================================
//--- mod_enforced.cppm
// expected-no-diagnostics
export module TestMod [[profiles::enforce(test::type_cast)]];

export void mod_func();

// ===================================================================
// Require on import: OK when designators match
// ===================================================================
//--- require_ok.cpp
// expected-no-diagnostics
import TestMod [[profiles::require(test::type_cast)]];

// ===================================================================
// Require on import: error when profile not enforced by module
// ===================================================================
//--- require_fail.cpp
import TestMod [[profiles::require(test::not_enforced)]]; // expected-error {{required profile 'test::not_enforced' is not enforced by imported module}}

// ===================================================================
// Require with designator mismatch
// ===================================================================
//--- require_mismatch.cpp
import TestMod [[profiles::require(test::type_cast(strict: true))]]; // expected-error {{required profile 'test::type_cast(strict : true)' is not enforced by imported module}}

// ===================================================================
// Repeated require of same designator: OK (P3589R2 Section 2.3 example)
// ===================================================================
//--- require_repeated.cpp
// expected-no-diagnostics
import TestMod [[profiles::require(test::type_cast)]];
import TestMod [[profiles::require(test::type_cast)]];

// ===================================================================
// Interface-to-implementation propagation: the implementation unit
// inherits the enforced profile from the module interface, so the
// reinterpret_cast check fires without a local enforce.
// ===================================================================
//--- impl_propagation.cpp
module TestMod;

void impl_func() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// ===================================================================
// Module with GMF enforce preceding module-declaration enforce:
// require must still find the profile on the module.
// ===================================================================
//--- mod_gmf_enforce.cppm
// expected-no-diagnostics
module;
[[profiles::enforce(test::type_cast)]];
export module GmfMod [[profiles::enforce(test::type_cast)]];

export void gmf_func();

//--- require_gmf_ok.cpp
// expected-no-diagnostics
import GmfMod [[profiles::require(test::type_cast)]];

// ===================================================================
// GMF-only enforce (no enforce on module-declaration): the profile is
// enforced in the TU but NOT exported via the module. A require on
// import must fail per P3589R2 [decl.attr.require]p2.
// ===================================================================
//--- mod_gmf_only_enforce.cppm
// expected-no-diagnostics
module;
[[profiles::enforce(test::type_cast)]];
export module GmfOnlyMod;

export void gmf_only_func();

//--- require_gmf_only_fail.cpp
import GmfOnlyMod [[profiles::require(test::type_cast)]]; // expected-error {{required profile 'test::type_cast' is not enforced by imported module}}
