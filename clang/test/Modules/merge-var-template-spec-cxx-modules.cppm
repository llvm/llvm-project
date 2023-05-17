// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/var_def.cppm -o %t/var_def.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/reexport1.cppm -o %t/reexport1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/reexport2.cppm -o %t/reexport2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cppm -fsyntax-only -verify

//--- use.cppm
import reexport1;
import reexport2;

auto foo = zero<Int>;
auto bar = zero<int*>;
auto baz = zero<int>;

template <class T> constexpr T zero = 0; // expected-error-re {{declaration{{.*}}in the global module follows declaration in module var_def}}
                                         // expected-note@* {{previous}}
template <> constexpr Int zero<Int> = {0}; // expected-error-re {{declaration{{.*}}in the global module follows declaration in module var_def}}
                                           // expected-note@* {{previous}}
template <class T> constexpr T* zero<T*> = nullptr; // expected-error-re {{declaration{{.*}}in the global module follows declaration in module var_def}}
                                                    // expected-note@* {{previous}}

template <> constexpr int** zero<int**> = nullptr; // ok, new specialization.
template <class T> constexpr T** zero<T**> = nullptr; // ok, new partial specilization.

//--- var_def.cppm
export module var_def;

export template <class T> constexpr T zero = 0;
export struct Int {
    int value;
};
export template <> constexpr Int zero<Int> = {0};
export template <class T> constexpr T* zero<T*> = nullptr;

//--- reexport1.cppm
export module reexport1;
export import var_def;

//--- reexport2.cppm
export module reexport2;
export import var_def;
