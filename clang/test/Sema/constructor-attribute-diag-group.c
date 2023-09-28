// RUN: %clang_cc1 -fsyntax-only -verify=err %s
// RUN: %clang_cc1 -fsyntax-only -verify=warn -Wno-error=priority-ctor-dtor %s
// RUN: %clang_cc1 -fsyntax-only -verify=okay -Wno-priority-ctor-dtor %s
// RUN: %clang_cc1 -fsyntax-only -verify=okay -Wno-prio-ctor-dtor %s
// okay-no-diagnostics

void f(void) __attribute__((constructor(1)));   // warn-warning {{'constructor' attribute requires integer constant between 101 and 65535 inclusive}} \
                                                   err-error {{'constructor' attribute requires integer constant between 101 and 65535 inclusive}}
void f(void) __attribute__((destructor(1)));    // warn-warning {{'destructor' attribute requires integer constant between 101 and 65535 inclusive}} \
                                                   err-error {{'destructor' attribute requires integer constant between 101 and 65535 inclusive}}
