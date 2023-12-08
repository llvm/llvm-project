// RUN: %clang_cc1 -verify -print-dependency-directives-minimized-source %s 2>&1

@import x // expected-error {{could not find ';' after @import}}
