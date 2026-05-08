// RUN: %clang_cc1 -fsyntax-only -verify=c   -x c   %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s

int x = (::h); // c-error {{expected expression}} \
                  cxx-error {{no member named 'h' in the global namespace}}
