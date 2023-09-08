// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#define CF_OPTIONS(_type, _name) __attribute__((availability(swift, unavailable))) _type _name; enum : _name

__attribute__((availability(macOS, unavailable)))
typedef CF_OPTIONS(unsigned, TestOptions) {
  x
};
