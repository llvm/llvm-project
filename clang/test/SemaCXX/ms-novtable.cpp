// RUN: %clang_cc1 -triple i386-pc-win32 %s -fsyntax-only -verify -fms-extensions -Wno-microsoft -std=c++11

struct __declspec(novtable) S {};
enum __declspec(novtable) E {}; // expected-warning{{'novtable' attribute only applies to classes}}
int __declspec(novtable) I; // expected-warning{{'novtable' attribute only applies to classes}}
typedef struct T __declspec(novtable) U; // expected-warning{{'novtable' attribute only applies to classes}}
auto z = []() __declspec(novtable) { return nullptr; }; // expected-warning{{'novtable' attribute only applies to classes}}
