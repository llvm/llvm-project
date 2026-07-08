// Header units (P3589R2 s2.3): [[profiles::enforce]] on an empty-declaration
// in a header compiled as a header unit is recorded on the header-unit module
// and validated by [[profiles::require]] on its import. Importing an enforced
// header unit does not enforce the profile locally.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -emit-header-unit -xc++-user-header %t/enforced.h -o %t/enforced.pcm
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -emit-header-unit -xc++-user-header %t/plain.h -o %t/plain.pcm
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -emit-header-unit -xc++-user-header %t/args.h -o %t/args.pcm
//
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_ok.cpp -fmodule-file=%t/enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_fail.cpp -fmodule-file=%t/enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_plain_fail.cpp -fmodule-file=%t/plain.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_no_leak.cpp -fmodule-file=%t/enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_args_ok.cpp -fmodule-file=%t/args.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/import_args_fail.cpp -fmodule-file=%t/args.pcm -verify
//
// The -fprofiles-test-profiles gate only controls whether test:: rules fire;
// the designator is recorded and exported regardless (see the Test Profiles
// section of ProfilesFrameworkInternals.rst), so require validates identically
// under plain -fprofiles.
// RUN: %clang_cc1 -std=c++20 -fprofiles -emit-header-unit -xc++-user-header %t/enforced.h -o %t/enforced-noflag.pcm
// RUN: %clang_cc1 -std=c++20 -fprofiles -Wno-experimental-header-units -fsyntax-only %t/import_ok.cpp -fmodule-file=%t/enforced-noflag.pcm -verify
//
// Redeclaration profile compatibility (P3589R2 [decl.attr.enforce]p5)
// across a header unit, in both directions.
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/redecl_forward.cpp -fmodule-file=%t/enforced.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fprofiles -fprofiles-test-profiles -Wno-experimental-header-units -fsyntax-only %t/redecl_reverse.cpp -fmodule-file=%t/plain.pcm -verify

//--- enforced.h
[[profiles::enforce(test::type_cast)]];
void hu_api(int);

//--- plain.h
void plain_api(int);

//--- args.h
[[profiles::enforce(vendor(fortify: 3))]];
void args_api(int);

//--- import_ok.cpp
// expected-no-diagnostics
import "enforced.h" [[profiles::require(test::type_cast)]];

//--- import_fail.cpp
import "enforced.h" [[profiles::require(test::other)]]; // expected-error {{required profile 'test::other' is not enforced by imported module}}

//--- import_plain_fail.cpp
// A header unit built with no enforcement satisfies no requirement.
import "plain.h" [[profiles::require(test::type_cast)]]; // expected-error {{required profile 'test::type_cast' is not enforced by imported module}}

//--- import_no_leak.cpp
// Importing an enforced header unit does not enforce the profile in the
// importer: enforcement is always explicit and local.
// expected-no-diagnostics
import "enforced.h";
long no_leak(void *p) { return reinterpret_cast<long>(p); }

//--- import_args_ok.cpp
// expected-no-diagnostics
import "args.h" [[profiles::require(vendor(fortify: 3))]];

//--- import_args_fail.cpp
// Require compares canonical designator spellings, arguments included.
import "args.h" [[profiles::require(vendor(fortify: 2))]]; // expected-error {{required profile 'vendor(fortify : 2)' is not enforced by imported module}}

//--- redecl_forward.cpp
// The header unit's TU enforced a profile; the redeclaring TU must enforce a
// compatible one.
import "enforced.h";
void hu_api(int); // expected-error {{redeclaration of 'hu_api' is not in the dominion of a profile compatible with 'test::type_cast'}}
// expected-note@enforced.h:* {{previous declaration is here}}

//--- redecl_reverse.cpp
// And symmetrically: this TU enforces a profile; the header unit's TU did not
// enforce a compatible one.
[[profiles::enforce(test::type_cast)]];
import "plain.h";
void plain_api(int); // expected-error {{'plain_api' was previously declared in module}}
// expected-note@plain.h:* {{previous declaration is here}}
