// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/implementation.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/interface.cpp
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -fno-caret-diagnostics %t/implementation.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IMPL-LOC
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -fno-caret-diagnostics %t/interface.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IFACE-LOC
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/same-line.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/private.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/partition.cpp
// RUN: %clang_cc1 -std=c++20 -E -P %t/implementation.cpp | FileCheck %s --check-prefix=PP
// RUN: %python -c "from pathlib import Path; [p.write_bytes(p.read_bytes().rstrip(b'\r\n')) for p in Path(r'%t').glob('eof-*.cpp')]"
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-identifier.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-colon.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-semicolon.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-export-identifier.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-export-colon.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/eof-export-semicolon.cpp

// Each declaration following a malformed directive is used immediately, so
// skipping that declaration during recovery produces an unexpected diagnostic.
// The final failing assertion also checks that recovery does not skip to EOF.
// Preprocessing must preserve the declarations on subsequent lines as well.

// Check that the implementation diagnostic points to the first invalid token.
// The interface diagnostic should point to the colon where a name is expected.
// IMPL-LOC: implementation.cpp:1:9: error: expected private
// IFACE-LOC: interface.cpp:1:15: error: expected module name

// PP: constexpr int after_identifier = 42;
// PP: constexpr int after_semicolon = 42;
// PP: constexpr int after_parentheses = 42;
// PP: constexpr int after_brackets = 42;
// PP: constexpr int after_braces = 42;
// PP: constexpr int after_dotted_name = 42;
// PP: constexpr int after_macro = 42;
// PP: constexpr int after_comment = 42;
// PP: constexpr int after_missing_semicolon = 42;
// PP: constexpr int after_colon = 42;
// PP: constexpr int after_unclosed_parenthesis = 42;

//--- implementation.cpp
module :foo; // expected-error {{expected private}}
constexpr int after_identifier = 42;
static_assert(after_identifier == 42);

module :; // expected-error {{expected private}}
constexpr int after_semicolon = 42;
static_assert(after_semicolon == 42);

module :(bad); // expected-error {{expected private}}
constexpr int after_parentheses = 42;
static_assert(after_parentheses == 42);

module :[bad]; // expected-error {{expected private}}
constexpr int after_brackets = 42;
static_assert(after_brackets == 42);

module :{bad}; // expected-error {{expected private}}
constexpr int after_braces = 42;
static_assert(after_braces == 42);

module :foo.bar; // expected-error {{expected private}}
constexpr int after_dotted_name = 42;
static_assert(after_dotted_name == 42);

#define BAD_NAME foo.bar
module :BAD_NAME; // expected-error {{expected private}}
constexpr int after_macro = 42;
static_assert(after_macro == 42);

module : /* comment */ foo; // expected-error {{expected private}}
constexpr int after_comment = 42;
static_assert(after_comment == 42);

module :foo // expected-error {{expected private}}
constexpr int after_missing_semicolon = 42;
static_assert(after_missing_semicolon == 42);

module : // expected-error {{expected private}}
constexpr int after_colon = 42;
static_assert(after_colon == 42);

module :( // expected-error {{expected private}}
constexpr int after_unclosed_parenthesis = 42;
static_assert(after_unclosed_parenthesis == 42);

static_assert(false, "recovery reached"); // expected-error {{static assertion failed: recovery reached}}

//--- interface.cpp
export module :foo; // expected-error {{expected module name}}
constexpr int after_identifier = 42;
static_assert(after_identifier == 42);

export module :; // expected-error {{expected module name}}
constexpr int after_semicolon = 42;
static_assert(after_semicolon == 42);

export module :(bad); // expected-error {{expected module name}}
constexpr int after_parentheses = 42;
static_assert(after_parentheses == 42);

export module :[bad]; // expected-error {{expected module name}}
constexpr int after_brackets = 42;
static_assert(after_brackets == 42);

export module :{bad}; // expected-error {{expected module name}}
constexpr int after_braces = 42;
static_assert(after_braces == 42);

export module :foo // expected-error {{expected module name}}
constexpr int after_missing_semicolon = 42;
static_assert(after_missing_semicolon == 42);

export module : // expected-error {{expected module name}}
constexpr int after_colon = 42;
static_assert(after_colon == 42);

export module :( // expected-error {{expected module name}}
constexpr int after_unclosed_parenthesis = 42;
static_assert(after_unclosed_parenthesis == 42);

static_assert(false, "recovery reached"); // expected-error {{static assertion failed: recovery reached}}

//--- same-line.cpp
// Tokens after the terminating semicolon are diagnosed by the preprocessor,
// but must still be available to the parser after recovery.
module :foo; constexpr int after_same_line = 42; // expected-error {{expected private}} expected-warning {{extra tokens after semicolon in 'module' directive}}
static_assert(after_same_line == 42);
static_assert(false, "recovery reached"); // expected-error {{static assertion failed: recovery reached}}

//--- private.cpp
// expected-no-diagnostics
export module M;
module :private;
constexpr int after_private = 42;
static_assert(after_private == 42);

//--- partition.cpp
// expected-no-diagnostics
export module M:part;
export constexpr int value = 42;
static_assert(value == 42);

//--- eof-identifier.cpp
// EOF cases are stripped of trailing newlines before compilation.
// expected-error@+1 {{expected private}}
module :foo
//--- eof-colon.cpp
// expected-error@+1 {{expected private}}
module :
//--- eof-semicolon.cpp
// expected-error@+1 {{expected private}}
module :;
//--- eof-export-identifier.cpp
// expected-error@+1 {{expected module name}}
export module :foo
//--- eof-export-colon.cpp
// expected-error@+1 {{expected module name}}
export module :
//--- eof-export-semicolon.cpp
// expected-error@+1 {{expected module name}}
export module :;
