// RUN: %clang_cc1 -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=none -pedantic
// RUN: %clang_cc1 -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=compat -Wpre-c23-compat
// RUN: %clang_cc1 -std=c17 %s -fsyntax-only --embed-dir=%S/Inputs -verify=ext -pedantic
// RUN: %clang_cc1 -x c++ -std=c++26 %s -fsyntax-only --embed-dir=%S/Inputs -verify=none -pedantic
// RUN: %clang_cc1 -x c++ -std=c++26 %s -fsyntax-only --embed-dir=%S/Inputs -verify=cxx-compat -Wpre-c++26-compat
// RUN: %clang_cc1 -x c++ -std=c++23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=cxx-ext -pedantic
// none-no-diagnostics

#if __has_embed("jk.txt")

const char buffer[] = {
#embed "jk.txt" /* compat-warning {{#embed is incompatible with C standards before C23}}
                   ext-warning {{#embed is a C23 extension}}
                   cxx-compat-warning {{#embed is incompatible with C++ standards before C++2c}}
                   cxx-ext-warning {{#embed is a C++2c extension}}
                 */
};
#endif
