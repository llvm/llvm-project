// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Invisible.cppm -emit-module-interface -o %t/Invisible.pcm
// RUN: %clang_cc1 -std=c++20 %t/Other.cppm -emit-module-interface -fprebuilt-module-path=%t \
// RUN:     -o %t/Other.pcm
// RUN: %clang_cc1 -std=c++20 %t/Another.cppm -emit-module-interface -o %t/Another.pcm
// RUN: %clang_cc1 -std=c++20 %t/A-interface.cppm -emit-module-interface \
// RUN:     -fprebuilt-module-path=%t -o %t/A-interface.pcm
// RUN: %clang_cc1 -std=c++20 %t/A-interface2.cppm -emit-module-interface \
// RUN:     -fprebuilt-module-path=%t -o %t/A-interface2.pcm
// RUN: %clang_cc1 -std=c++20 %t/A-interface3.cppm -emit-module-interface \
// RUN:     -fprebuilt-module-path=%t -o %t/A-interface3.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface \
// RUN:     -fprebuilt-module-path=%t -o %t/A.pcm

// RUN: %clang_cc1 -std=c++20 %t/A.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/A-impl.cppm -fprebuilt-module-path=%t -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/A-impl2.cppm -fprebuilt-module-path=%t -fsyntax-only -verify

//--- Invisible.cppm
export module Invisible;
export void invisible() {}

//--- Other.cppm
export module Other;
import Invisible;
export void other() {}

//--- Another.cppm
export module Another;
export void another() {}

//--- A-interface.cppm
export module A:interface;
import Other;
export void a_interface() {}

//--- A-interface2.cppm
export module A:interface2;
import Another;
export void a_interface2() {}

//--- A-interface3.cppm
export module A:interface3;
import :interface;
import :interface2;
export void a_interface3() {}

//--- A.cppm
export module A;
import Another;
import :interface;
import :interface2;
import :interface3;

export void a() {}
export void impl();

//--- A.cpp
module A;
void impl() {
    a_interface();
    a_interface2();
    a_interface3();

    other();
    another();

    invisible(); // expected-error {{declaration of 'invisible' must be imported from module 'Invisible' before it is required}}
                 // expected-note@* {{declaration here is not visible}}
}

//--- A-impl.cppm
module A:impl;
import :interface3;

void impl_part() {
    a_interface();
    a_interface2();
    a_interface3();

    other();
    another();

    invisible(); // expected-error {{declaration of 'invisible' must be imported from module 'Invisible' before it is required}}
                 // expected-note@* {{declaration here is not visible}}
}

//--- A-impl2.cppm
module A:impl2;
import A;

void impl_part2() {
    a();
    impl();

    a_interface();
    a_interface2();
    a_interface3();

    other();
    another();

    invisible(); // expected-error {{declaration of 'invisible' must be imported from module 'Invisible' before it is required}}
                 // expected-note@* {{declaration here is not visible}}
}
