// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang -std=c++20 %S/Inputs/module-transtive-instantiation/Templ.cppm --precompile -o %t/Templ.pcm
// RUN: %clang -std=c++20 %S/Inputs/module-transtive-instantiation/bar.cppm  --precompile -fprebuilt-module-path=%t -o %t/bar.pcm
// RUN: %clang -std=c++20 -fprebuilt-module-path=%t %s -c -Xclang -verify

import bar;
int foo() {
    // FIXME: It shouldn't be an error. Since the `G` is already imported in bar.
    return bar<int>(); // expected-error@Inputs/module-transtive-instantiation/bar.cppm:5 {{definition of 'G' must be imported from module 'Templ' before it is required}}
                       // expected-note@-1 {{in instantiation of function template specialization 'bar<int>' requested here}}
                       // expected-note@Inputs/module-transtive-instantiation/Templ.cppm:3 {{definition here is not reachable}}
}
