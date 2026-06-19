// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_enforced.cppm -o %t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_ok.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_fail.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_mismatch.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_repeated.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -fsyntax-only %t/impl_propagation.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_gmf_enforce.cppm -o %t/mod_gmf_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_gmf_ok.cpp -fmodule-file=GmfMod=%t/mod_gmf_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_gmf_only_enforce.cppm -o %t/mod_gmf_only_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/require_gmf_only_fail.cpp -fmodule-file=GmfOnlyMod=%t/mod_gmf_only_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/part_iface.cppm -o %t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_primary_require_ok.cppm -fmodule-file=PartMod:part=%t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_primary_require_fail.cppm -fmodule-file=PartMod:part=%t/part_iface.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -fsyntax-only %t/part_iface_violation.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -fsyntax-only %t/part_impl_enforce.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -fsyntax-only %t/part_impl_inherit.cppm -fmodule-file=%t/mod_enforced.pcm -Wno-eager-load-cxx-named-modules -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/part_impl_no_inherit.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_no_local_enforce.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_gmf_only_no_leak.cpp -fmodule-file=GmfOnlyMod=%t/mod_gmf_only_enforce.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-module-interface %t/mod_different_desig.cppm -o %t/mod_different_desig.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_two_modules.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -fmodule-file=DiffDesigMod=%t/mod_different_desig.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/mod_noflag_enforce.cppm -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/mod_bare.cppm -o %t/mod_bare.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/import_noflag_require.cpp -fmodule-file=BareMod=%t/mod_bare.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/mod_enforce_no_args.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fsyntax-only %t/import_require_no_args.cpp -fmodule-file=TestMod=%t/mod_enforced.pcm -verify

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
// Partition implementation inherits the primary interface's enforced
// profiles when that interface's BMI is resident (best-effort). Only the
// eager -fmodule-file=<path> form makes TestMod resident here; the lazy
// -fmodule-file=<name>=<path> form loads on import only and would not
// trigger inheritance (the partition impl does not import the primary).
// TestMod enforces test::type_cast, so the cast is diagnosed without a
// local enforce.
// ===================================================================
//--- part_impl_inherit.cppm
module TestMod:inherit;

void part_inherit_func() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// ===================================================================
// Normal build: the primary interface is not implicitly imported and is
// usually built after its partitions, so its BMI is absent here and the
// partition implementation unit does NOT inherit the enforcement. This is
// the best-effort limitation; repeat [[profiles::enforce]] for guaranteed
// enforcement.
// ===================================================================
//--- part_impl_no_inherit.cppm
// expected-no-diagnostics
module TestMod:inherit;

void part_no_inherit_func() {
  int *p = reinterpret_cast<int*>(0);
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

// ===================================================================
// Without -fprofiles, [[profiles::enforce]] on a module-declaration
// must emit warn_attribute_ignored instead of being silently accepted.
// ===================================================================
//--- mod_noflag_enforce.cppm
export module NoFlagMod [[profiles::enforce(test::type_cast)]]; // expected-warning {{'profiles::enforce' attribute ignored}}

export void f();

// ===================================================================
// A plain module with no profile attrs, built without -fprofiles, to
// serve as an import target for the require-without-flag test below.
// ===================================================================
//--- mod_bare.cppm
// expected-no-diagnostics
export module BareMod;

export void bare_fn();

// ===================================================================
// Without -fprofiles, [[profiles::require]] on an import must emit
// warn_attribute_ignored and must NOT produce the spurious
// err_profiles_require_not_enforced diagnostic.
// ===================================================================
//--- import_noflag_require.cpp
import BareMod [[profiles::require(test::type_cast)]]; // expected-warning {{'profiles::require' attribute ignored}}

// ===================================================================
// [[profiles::enforce]] on a module-declaration with no argument
// clause must be diagnosed, not crash.
// ===================================================================
//--- mod_enforce_no_args.cppm
export module NoArgsMod [[profiles::enforce]]; // expected-error {{'enforce' attribute requires an argument clause}}

export void f();

// ===================================================================
// [[profiles::require]] on an import-declaration with no argument
// clause must be diagnosed, not crash.
// ===================================================================
//--- import_require_no_args.cpp
import TestMod [[profiles::require]]; // expected-error {{'require' attribute requires an argument clause}}
