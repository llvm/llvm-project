// RUN: %clang_cc1 -fsyntax-only -verify -fprofiles -std=c++20 %s

// ===================================================================
// Valid enforce forms
// ===================================================================

[[profiles::enforce(std::type)]];

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
// Parse errors
// ===================================================================

// enforce with empty parens: parse error
[[profiles::enforce()]]; // expected-error {{expected profile name}}

// ===================================================================
// Profile name with :: separators
// ===================================================================

[[profiles::suppress(a::b::c)]]
void deep_name();
