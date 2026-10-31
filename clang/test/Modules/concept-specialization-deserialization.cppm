// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=a=%t/a.pcm -fsyntax-only %t/use.cpp -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=a=%t/a.pcm -fsyntax-only %t/use.cpp -verify

//--- a.cppm
export module a;
template <class> concept C = true;

template <class T> int fn() noexcept(C<T>);
export using t = decltype(fn<int>());

//--- use.cpp
// expected-no-diagnostics
import a;

// During the deserialization process of fn<int>, the C<int> in noexcept expression
// may be not completely deserialized. This test makes sure that we can handle the case.
//
// The ordering is:
//
//  Deserialize C<int>
//
//  Deserializing C<int>'s template argument
//
//  Read SubstTemplateTypeParmType::AssociatedDecl in readSubstTemplateTypeParmType
//
//  Deserialize fn<int>
//
//  FunctionProtoType::Profile
//
//  but C<int>'s template arguments is not deserialized yet.
t x;
