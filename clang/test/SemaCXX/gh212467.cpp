// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

[[gnu::warning("same")]]
[[gnu::warning("same")]]
void same_warning();

[[gnu::warning("one")]] // expected-note {{previous attribute is here}}
[[gnu::warning("two")]] // expected-warning {{attribute 'gnu::warning' is already applied with different arguments}}
void different_warning();

[[gnu::error("same")]]
[[gnu::error("same")]]
void same_error();

[[gnu::error("one")]]   // expected-error {{'gnu::warning' and 'gnu::error' attributes are not compatible}}
[[gnu::warning("two")]] // expected-note {{conflicting attribute is here}}
void conflicting();

[[gnu::warning("same")]]
__attribute__((warning("same"))) void mixed_syntax();

__attribute__((warning("same")))
[[gnu::warning("same")]] void reverse_mixed_syntax();
