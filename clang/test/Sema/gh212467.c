// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// expected-no-diagnostics

[[gnu::warning("same")]]
[[gnu::warning("same")]]
void same_warning(void);

[[gnu::error("same")]]
[[gnu::error("same")]]
void same_error(void);

[[gnu::warning("same")]]
__attribute__((warning("same"))) void mixed_warning(void);

__attribute__((error("same")))
[[gnu::error("same")]] void reverse_mixed_error(void);
