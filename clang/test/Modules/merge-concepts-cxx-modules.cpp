// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/same_as.cppm -o %t/same_as.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/concepts.cppm -o %t/concepts.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/format.cppm -o %t/format.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/conflicting.cppm -o %t/conflicting.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cppm -fsyntax-only -verify

//--- same_as.cppm
export module same_as;
export template <class T, class U>
concept same_as = __is_same(T, U);

//--- concepts.cppm
export module concepts;
export import same_as;

export template <class T>
concept ConflictingConcept = true;

//--- format.cppm

export module format;
export import concepts;
export import same_as;

export template <class T> void foo()
  requires same_as<T, int>
{}

//--- conflicting.cppm
export module conflicting;
export template <class T, class U = int>
concept ConflictingConcept = true;

//--- Use.cppm
import format;
import conflicting;

template <class T> void foo()
  requires same_as<T, T>
{}
ConflictingConcept auto x = 10; // expected-error {{reference to 'ConflictingConcept' is ambiguous}}
                                // expected-note@* 2 {{candidate}}
