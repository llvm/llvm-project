// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/module.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/module.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/import.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/import.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/_Test.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/_Test.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/__test.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/__test.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/te__st.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/te__st.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/std.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std.foo.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/std.foo.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std0.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/std0.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std1000000.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/std1000000.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/module.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/module.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/import.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/import.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/_Test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/_Test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/__test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/__test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/te__st.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/te__st.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/std.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std.foo.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/std.foo.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std0.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/std0.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/std1000000.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/std1000000.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/should_diag._Test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -DNODIAGNOSTICS -Wno-reserved-module-identifier %t/should_diag._Test.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/system-module.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/system-module.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/system._Test.import.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/system._Test.import.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,loud %t/user.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected -Wno-reserved-module-identifier %t/user.cpp

//--- module.cpp
module module;  // expected-error {{'module' is an invalid name for a module}}

//--- import.cpp
module import;  // expected-error {{'import' is an invalid name for a module}}

//--- _Test.cpp
module _Test;   // loud-warning {{'_Test' is a reserved name for a module}}  \
                // expected-error {{module '_Test' not found}}

//--- __test.cpp
module __test; // loud-warning {{'__test' is a reserved name for a module}} \
               // expected-error {{module '__test' not found}}

//--- te__st.cpp
module te__st; // loud-warning {{'te__st' is a reserved name for a module}} \
               // expected-error {{module 'te__st' not found}}

//--- std.cpp
module std; // loud-warning {{'std' is a reserved name for a module}} \
            // expected-error {{module 'std' not found}}

//--- std.foo.cpp
module std.foo; // loud-warning {{'std' is a reserved name for a module}} \
                // expected-error {{module 'std.foo' not found}}

//--- std0.cpp
module std0; // loud-warning {{'std0' is a reserved name for a module}} \
             // expected-error {{module 'std0' not found}}

//--- std1000000.cpp
module std1000000; // loud-warning {{'std1000000' is a reserved name for a module}} \
                   // expected-error {{module 'std1000000' not found}}

//--- module.cppm
export module module; // expected-error {{'module' is an invalid name for a module}}

//--- import.cppm
export module import; // expected-error {{'import' is an invalid name for a module}}

//--- _Test.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module _Test;  // loud-warning {{'_Test' is a reserved name for a module}}

//--- __test.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module __test; // loud-warning {{'__test' is a reserved name for a module}}
export int a = 43;

//--- te__st.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module te__st; // loud-warning {{'te__st' is a reserved name for a module}}
export int a = 43;

//--- std.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module std;    // loud-warning {{'std' is a reserved name for a module}}

export int a = 43;

//--- std.foo.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module std.foo;// loud-warning {{'std' is a reserved name for a module}}

//--- std0.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module std0;   // loud-warning {{'std0' is a reserved name for a module}}

//--- std1000000.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module std1000000; // loud-warning {{'std1000000' is a reserved name for a module}}

//--- should_diag._Test.cppm
#ifdef NODIAGNOSTICS
// expected-no-diagnostics
#endif
export module should_diag._Test; // loud-warning {{'_Test' is a reserved name for a module}}

//--- system-module.cppm
// Show that being in a system header doesn't save you from diagnostics about
// use of an invalid module-name identifier.
# 34 "reserved-names-1.cpp" 1 3
export module module;       // expected-error {{'module' is an invalid name for a module}}

//--- system._Test.import.cppm
# 34 "reserved-names-1.cpp" 1 3
export module _Test.import; // expected-error {{'import' is an invalid name for a module}}

//--- user.cpp
// We can still use a reserved name on imoport.
import std; // expected-error {{module 'std' not found}}
