// RUN: %clang_cc1 -std=c++11 -triple i386-windows-msvc \
// RUN:   -aux-triple nvptx-nvidia-cuda -fsyntax-only -verify %s

// RUN: %clang_cc1 -std=c++11 -triple nvptx-nvidia-cuda \
// RUN:   -aux-triple i386-windows-msvc -fsyntax-only \
// RUN:   -fcuda-is-device -verify %s

// RUN: %clang_cc1 -std=c++11 -triple nvptx-nvidia-cuda \
// RUN:   -aux-triple x86_64-linux-gnu -fsyntax-only \
// RUN:   -fcuda-is-device -verify -verify-ignore-unexpected=note \
// RUN:   -DEXPECT_ERR %s

// CUDA device code should inherit the host's calling conventions.

template <class T>
struct Foo;

template <class T>
struct Foo<T()> {};

// On x86_64-linux-gnu, this is a redefinition of the template, because the
// __fastcall calling convention doesn't exist (and is therefore ignored).
#ifndef EXPECT_ERR
// expected-no-diagnostics
#else
// expected-error@+4 {{redefinition of 'Foo}}
// expected-warning@+3 {{'__fastcall' calling convention is not supported}}
#endif
template <class T>
struct Foo<T __fastcall()> {};
