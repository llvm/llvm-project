// RUN: %clang_cc1 -triple i386-pc-win32 %s -fsyntax-only -verify -fms-extensions -Wno-microsoft -std=c++11

struct __declspec(layout_version(19)) S {};
enum __declspec(layout_version(19)) E {}; // expected-warning{{'layout_version' attribute only applies to classes}}
int __declspec(layout_version(19)) I; // expected-warning{{'layout_version' attribute only applies to classes}}
typedef struct T __declspec(layout_version(19)) U; // expected-warning{{'layout_version' attribute only applies to classes}}
auto z = []() __declspec(layout_version(19)) { return nullptr; }; // expected-warning{{'layout_version' attribute only applies to classes}}

struct __declspec(layout_version(18)) X {}; // expected-error{{'layout_version' attribute parameter 18 is out of bounds}}
struct __declspec(layout_version(20)) Y {}; // expected-error{{'layout_version' attribute parameter 20 is out of bounds}}
struct __declspec(layout_version) Z {}; // expected-error{{attribute takes one argument}}
