// RUN: %clang_cc1 -triple aarch64 -verify -fsyntax-only %s

__attribute__((target("branch-protection=foo"))) // expected-error {{invalid or misplaced branch protection specification 'foo'}}
void
badvalue0() {}

__attribute__((target("branch-protection=+bti"))) // expected-error {{invalid or misplaced branch protection specification '<empty>'}}
void
badvalue1() {}

__attribute__((target("branch-protection=bti+"))) // expected-error {{invalid or misplaced branch protection specification '<empty>'}}
void
badvalue2() {}

__attribute__((target("branch-protection=pac-ret+bkey"))) // expected-error {{invalid or misplaced branch protection specification 'bkey'}}
void
badvalue3() {}

__attribute__((target("branch-protection=bti+leaf"))) // expected-error {{invalid or misplaced branch protection specification 'leaf'}}
void
badoption0() {}

__attribute__((target("branch-protection=bti+leaf+pac-ret"))) // expected-error {{invalid or misplaced branch protection specification 'leaf'}}
void
badorder0() {}

__attribute__((target("branch-protection=pac-ret+bti+leaf"))) // expected-error {{invalid or misplaced branch protection specification 'leaf'}}
void
badorder1() {}
