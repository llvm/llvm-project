// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_enforced.cppm -o %t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_ok.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_fail.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_mismatch.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify

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
