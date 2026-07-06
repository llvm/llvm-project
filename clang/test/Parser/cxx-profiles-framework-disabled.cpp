// RUN: %clang_cc1 -fsyntax-only -verify=disabled -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=enabled -fprofiles -DPROFILES_ENABLED -std=c++23 %s

// With -fprofiles off, clang does not act on the profiles attributes: they
// are ignored with a warning like any standard attribute the implementation
// does not implement, and their argument clauses are ordinary balanced-token
// sequences, not checked against the P3589R2 profile grammar (which governs
// an implementation that enforces profiles). With -fprofiles on, the grammar
// errors fire as usual; this file asserts both behaviors side by side.

// Well-formed spellings: ignored when off, accepted when on.
// disabled-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::type)]];
// disabled-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(vendor::checks(fortify: 3))]];

// Malformed argument clauses: accepted as balanced-token soup when off,
// grammar errors when on.
// disabled-warning@+2 {{'profiles::enforce' attribute ignored}}
// enabled-error@+1 {{expected profile name}}
[[profiles::enforce(+)]];
// disabled-warning@+2 {{'profiles::enforce' attribute ignored}}
// enabled-error@+1 {{expected profile name}}
[[profiles::enforce()]];
// disabled-warning@+2 {{'profiles::enforce' attribute ignored}}
// enabled-error@+1 {{expected ')' in 'enforce' attribute}}
[[profiles::enforce(a b)]];

// A missing argument clause is likewise fine when off.
// disabled-warning@+2 {{'profiles::enforce' attribute ignored}}
// enabled-error@+1 {{'enforce' attribute requires an argument clause}}
[[profiles::enforce]];
// disabled-warning@+2 {{'profiles::require' attribute ignored}}
// enabled-error@+1 {{'require' attribute requires an argument clause}}
[[profiles::require]];

#ifndef PROFILES_ENABLED
// Semicolons, nested parens, and braces are all balanced tokens, fine in an
// ignored attribute's argument clause.
// disabled-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(a; (b), {c})]];
#endif

// disabled-warning@+2 {{'profiles::suppress' attribute ignored}}
// enabled-error@+1 {{'justification' argument of 'profiles::suppress' must be a string literal}}
[[profiles::suppress(p, justification: 42)]] int x = 0;
// disabled-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(std::type)]] int y = 0;

#ifndef PROFILES_ENABLED
// A genuinely unbalanced argument clause is not a valid attribute in any
// mode: the balanced-token skip still diagnoses the missing ')'.
// disabled-error@+4 {{expected ')'}}
// disabled-note@+3 {{to match this '('}}
// disabled-error@+2 {{expected ']'}}
// disabled-error@+2 {{expected external declaration}}
[[profiles::enforce(]];
#endif
