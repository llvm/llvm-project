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
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/part_iface.cppm -o %t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_primary_require_ok.cppm -fmodule-file=PartMod:part=%t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_primary_require_fail.cppm -fmodule-file=PartMod:part=%t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_iface_violation.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_impl_enforce.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_no_local_enforce.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_gmf_only_no_leak.cpp -fmodule-file=GmfOnlyMod=%t/mod_gmf_only_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_different_desig.cppm -o %t/mod_different_desig.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_two_modules.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -fmodule-file=DiffDesigMod=%t/mod_different_desig.pcm -verify

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

// ===================================================================
// Partition interface with enforce: the profile is exported via the
// partition module, so require on partition import succeeds, and
// enforcement fires locally within the partition.
// ===================================================================
//--- part_iface.cppm
// expected-no-diagnostics
export module PartMod:part [[profiles::enforce(test::type_cast)]];

export void part_func();

// ===================================================================
// Partition interface: enforcement fires locally within the partition.
// ===================================================================
//--- part_iface_violation.cppm
export module PartViol:part [[profiles::enforce(test::type_cast)]];

export void part_func() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

//--- part_primary_require_ok.cppm
// expected-no-diagnostics
export module PartMod;
import :part [[profiles::require(test::type_cast)]];

// ===================================================================
// Partition interface: require fails for a profile the partition does
// not enforce.
// ===================================================================
//--- part_primary_require_fail.cppm
export module PartMod;
import :part [[profiles::require(test::not_enforced)]]; // expected-error {{required profile 'test::not_enforced' is not enforced by imported module}}

// ===================================================================
// Partition implementation with enforce: enforcement is active locally
// but the profile is NOT exported on the module (ExportMod is null
// for PartitionImplementation).
// ===================================================================
//--- part_impl_enforce.cppm
module PartImpl:impl [[profiles::enforce(test::type_cast)]];

void impl_func() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// ===================================================================
// Importing a module with enforce must NOT leak enforcement into the
// importer's own TU. Without a local enforce, reinterpret_cast is OK.
// ===================================================================
//--- import_no_local_enforce.cpp
// expected-no-diagnostics
import TestMod;

void importer_func() {
  int *p = reinterpret_cast<int*>(0);
}

// ===================================================================
// GMF-only enforce must NOT leak into importers.
// ===================================================================
//--- import_gmf_only_no_leak.cpp
// expected-no-diagnostics
import GmfOnlyMod;

void gmf_importer_func() {
  int *p = reinterpret_cast<int*>(0);
}

// ===================================================================
// Second module enforcing test::type_cast with a different designator.
// ===================================================================
//--- mod_different_desig.cppm
// expected-no-diagnostics
export module DiffDesigMod [[profiles::enforce(test::type_cast(strict: true))]];

export void diff_func();

// ===================================================================
// Importing two modules that enforce the same profile name with
// different designators must NOT produce err_profiles_enforce_mismatch
// in the importer.
// ===================================================================
//--- import_two_modules.cpp
// expected-no-diagnostics
import TestMod;
import DiffDesigMod;
