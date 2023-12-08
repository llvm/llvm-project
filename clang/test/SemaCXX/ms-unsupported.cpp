// RUN: %clang_cc1 -triple x86_64-pc-windows-gnu %s -fsyntax-only -verify -fms-extensions -Wno-microsoft -std=c++11

// "novtable" is ignored except with the Microsoft C++ ABI.
// MinGW uses the Itanium C++ ABI so check that it is ignored there.
struct __declspec(novtable) S {}; // expected-warning{{__declspec attribute 'novtable' is not supported}}
