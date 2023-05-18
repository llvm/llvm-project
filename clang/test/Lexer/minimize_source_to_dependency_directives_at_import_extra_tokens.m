// RUN: %clang_cc1 -verify -print-dependency-directives-minimized-source %s 2>&1

@import x; a // expected-error {{unexpected extra tokens at end of @import declaration}}
