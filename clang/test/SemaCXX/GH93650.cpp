// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s
// expected-no-diagnostics

namespace std {
struct type_info {
  const char *name;
};
} // namespace std

namespace GH93650_bug {
auto func(auto... inputArgs) { return typeid(inputArgs...[0]); }
} // namespace GH93650_bug
