// REQUIRES: systemz-registered-target
// RUN: not %clang_cc1 -triple s390x-none-zos -fzos-extensions %s
__attribute__((visibility("hidden"))) int _Export i; // expected-error {{visibility does not match previous declaration}}
class __attribute__((visibility("hidden"))) _Export C; // expected-error {{visibility does not match previous declaration}}

