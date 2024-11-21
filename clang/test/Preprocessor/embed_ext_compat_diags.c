// RUN: %clang_cc1 -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=none -pedantic
// RUN: %clang_cc1 -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=compat -Wpre-c23-compat
// RUN: %clang_cc1 -std=c17 %s -fsyntax-only --embed-dir=%S/Inputs -verify=ext -pedantic
// RUN: %clang_cc1 -x c++ %s -fsyntax-only --embed-dir=%S/Inputs -verify=cxx -pedantic
// none-no-diagnostics

#if __has_embed("jk.txt")

const char buffer[] = {
#embed "jk.txt" /* compat-warning {{#embed is incompatible with C standards before C23}}
                   ext-warning {{#embed is a C23 extension}}
                   cxx-warning {{#embed is a Clang extension}}
                 */
};
#endif

