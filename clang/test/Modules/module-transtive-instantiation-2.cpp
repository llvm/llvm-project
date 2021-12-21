// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang -std=c++20 %S/Inputs/module-transtive-instantiation/Templ.cppm --precompile -o %t/Templ.pcm
// RUN: %clang -std=c++20 %S/Inputs/module-transtive-instantiation/bar.cppm  --precompile -fprebuilt-module-path=%t -o %t/bar.pcm
// RUN: %clang -std=c++20 -fprebuilt-module-path=%t %s -c -Xclang -verify

import bar;
int foo() {
    G<int> g;    // expected-error {{declaration of 'G' must be imported from module 'Templ' before it is required}}
    return g();  // expected-note@Inputs/module-transtive-instantiation/Templ.cppm:3 {{declaration here is not visible}}
}
