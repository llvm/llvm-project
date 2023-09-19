// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s -Wpre-c2x-compat
// RUN: %clang_cc1 -fsyntax-only -verify=c17 -std=c17 %s


constexpr int a = 0; // c17-error {{unknown type name 'constexpr'}} \
                        c23-warning {{'constexpr' is incompatible with C standards before C23}}
