// RUN: %clang_cc1 -triple i386-pc-win32 %s -fsyntax-only -verify -fms-extensions -Wno-microsoft -std=c++11

struct __declspec(empty_bases) S {};
enum __declspec(empty_bases) E {}; // expected-warning{{'empty_bases' attribute only applies to classes}}
int __declspec(empty_bases) I; // expected-warning{{'empty_bases' attribute only applies to classes}}
typedef struct T __declspec(empty_bases) U; // expected-warning{{'empty_bases' attribute only applies to classes}}
auto z = []() __declspec(empty_bases) { return nullptr; }; // expected-warning{{'empty_bases' attribute only applies to classes}}

struct __declspec(empty_bases(1)) X {}; // expected-error{{'empty_bases' attribute takes no arguments}}
