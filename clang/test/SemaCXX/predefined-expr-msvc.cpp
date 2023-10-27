// RUN: %clang_cc1 %s -std=c++17 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++17 -fsyntax-only -verify -fms-extensions

// expected-no-diagnostics

struct StringRef {
  StringRef(const char *);
};
template <typename T>
StringRef getTypeName() {
  StringRef s = __func__;
}

