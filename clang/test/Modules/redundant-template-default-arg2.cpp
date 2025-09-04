// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -I%t -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -I%t -emit-reduced-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -fprebuilt-module-path=%t -std=c++20 %t/use.cpp -fsyntax-only -verify

//--- foo.cppm
export module foo;
export template <typename T = int>
T v;

export template <int T = 8>
int v2;

export template <typename T>
class my_array {};

export template <template <typename> typename C = my_array>
int v3;

//--- use.cpp
import foo;
template <typename T = int>
T v; // expected-error {{declaration of 'v' in the global module follows declaration in module foo}}
     // expected-note@foo.cppm:3 {{previous declaration is here}}

template <int T = 8>
int v2; // expected-error {{declaration of 'v2' in the global module follows declaration in module foo}}
        // expected-note@foo.cppm:6 {{previous declaration is here}}

template <typename T>
class my_array {}; // expected-error {{declaration of 'my_array' in the global module follows declaration in module foo}}
                   // expected-note@foo.cppm:9 {{previous declaration is here}}

template <template <typename> typename C = my_array>
int v3; // expected-error {{declaration of 'v3' in the global module follows declaration in module foo}}
        // expected-note@foo.cppm:12 {{previous declaration is here}}
