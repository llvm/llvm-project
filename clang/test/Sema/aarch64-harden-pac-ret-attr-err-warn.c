// RUN: %clang_cc1 -triple aarch64 -verify -fsyntax-only %s

__attribute__((target("harden-pac-ret=none"))) // expected-warning {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void
badvalue0(void) {}

__attribute__((target("harden-pac-ret=load-return-address"))) // expected-warning {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void
badvalue1(void) {}

__attribute__((target("branch-protection=bti,harden-pac-ret=none"))) // expected-warning {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void
badvalue2(void) {}

__attribute__((target("branch-protection=bti,harden-pac-ret=load-return-address"))) // expected-warning {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void
badvalue3(void) {}

__attribute__((target("branch-protection=bti,harden-pac-ret=inexistent"))) // expected-error {{invalid or misplaced pac-ret hardening specification 'inexistent'}}
void
badvalue4(void) {}
