// RUN: %clang_cc1 -fsyntax-only -std=c2x -Wpre-c2x-compat -verify=pre-c2x %s
// RUN: %clang_cc1 -fsyntax-only -std=c17 -Wc2x-extensions -verify=c2x-ext %s

[[]] void func(); // pre-c2x-warning {{[[]] attributes are incompatible with C standards before C2x}}
                  // c2x-ext-warning@-1 {{[[]] attributes are a C2x extension}}
