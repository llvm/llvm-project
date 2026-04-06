// RUN: %clang_cc1 -fsyntax-only -verify -fprofiles -std=c++20 %s

// ===================================================================
// Valid enforce forms
// ===================================================================

// Single designator
[[profiles::enforce(std::type)]];

// Multi-designator in one enforce
[[profiles::enforce(acme::hardened, lib::safety)]];

// Designator with profile arguments
[[profiles::enforce(vendor(fortify: 3, sanitize: thread))]];

// Nested balanced groups in arguments
[[profiles::enforce(nested(config: (a b)))]];

// Bare non-operator-non-punctuator token arguments
[[profiles::enforce(bare1(3))]];

// Bare string literal argument
[[profiles::enforce(bare2("hello"))]];

// Bare identifier argument
[[profiles::enforce(bare3(abc))]];

// ===================================================================
// Valid suppress forms
// ===================================================================

[[profiles::suppress(std::type)]]
void suppress_no_args();

[[profiles::suppress(std::type, justification: "legacy")]]
void suppress_with_justification();

[[profiles::suppress(std::type, rule: "reinterpret_cast")]]
void suppress_with_rule();

[[profiles::suppress(std::type, justification: "legacy", rule: "cast")]]
void suppress_with_both();

// ===================================================================
// [[using profiles: ...]] syntax
// ===================================================================

[[using profiles: suppress(std::type)]]
void using_syntax();

// ===================================================================
// Profile name with :: separators
// ===================================================================

[[profiles::suppress(a::b::c)]]
void deep_name();

// ===================================================================
// Parse errors
// ===================================================================

// enforce with empty parens: parse error
[[profiles::enforce()]]; // expected-error {{expected profile name}}

// enforce with non-identifier first token
[[profiles::enforce(42)]]; // expected-error {{expected profile name}}

// suppress with empty parens: parse error
[[profiles::suppress()]]; // expected-error {{expected profile name}}

// Bare argument cannot be an operator (suppress uses profile-argument-list
// directly after comma, avoiding cascading errors from nested designator parens)
[[profiles::suppress(std::type, +)]] // expected-error {{invalid token in profile argument}}
void suppress_bare_operator();

// Bare argument cannot be a balanced group
[[profiles::suppress(std::type, (a b))]] // expected-error {{invalid token in profile argument}}
void suppress_bare_group();
