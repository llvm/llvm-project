// RUN: %check_clang_tidy -std=c++17-or-later %s readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-magic-numbers.IgnoredIntegerValues: "1;2;3;4;", \
// RUN:   readability-magic-numbers.IgnorePowersOf2IntegerValues: false}}' \
// RUN: --
//
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=,ALL %s \
// RUN: readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-magic-numbers.IgnoredIntegerValues: "1;2;3;4;", \
// RUN:   readability-magic-numbers.IgnorePowersOf2IntegerValues: false, \
// RUN:   readability-magic-numbers.IgnoreWellKnownFunctionArgs: false}}' \
// RUN: --

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <string>

void CFamily(const char *Str, const wchar_t *WStr) {
  (void)strtol(Str, nullptr, 0);
  (void)strtol(Str, nullptr, 8);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:30: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtol(Str, nullptr, 10);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:30: warning: 10 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtol(Str, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:30: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtoll(Str, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtoul(Str, nullptr, 8);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtoull(Str, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)wcstol(WStr, nullptr, 8);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)wcstoul(WStr, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void Qualified(const char *Str) {
  (void)std::strtol(Str, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:35: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)std::strtoull(Str, nullptr, 8);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:37: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void Cxx(const std::string &S, char *Begin, char *End, int &Value) {
  (void)std::stoi(S, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)std::stol(S, nullptr, 8);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)std::stoul(S, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  // The base is the fourth argument for the charconv functions.
  (void)std::from_chars(Begin, End, Value, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:44: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)std::to_chars(Begin, End, Value, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:42: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  // And the only argument here.
  (void)std::setbase(16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:22: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void Wrappers(const char *Str) {
  // Parentheses and the implicit conversion to int do not hide the argument.
  (void)strtol(Str, nullptr, (16));
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtol(Str, nullptr, ((8)));
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:32: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

template <typename T>
long ParseAs(const char *Str) {
  return strtol(Str, nullptr, 16);
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:31: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

long InstantiatedParse(const char *Str) { return ParseAs<int>(Str); }

namespace mylib {
long strtol(const char *Str, char **End, int Base);
}

void configure(const char *Str, char **End, int Flags);

void UnacceptedBase(const char *Str) {
  (void)strtol(Str, nullptr, 12);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 12 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void ComputedBase(const char *Str, bool Cond) {
  (void)strtol(Str, nullptr, 16 * 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)strtol(Str, nullptr, Cond ? 8 : 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 8 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  // CHECK-MESSAGES: :[[@LINE-2]]:41: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void WrongArgumentIndex(char *Begin, char *End) {
  (void)std::to_chars(Begin, End, 16, 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  // CHECK-MESSAGES-ALL: :[[@LINE-2]]:39: warning: 10 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void UnrelatedFunction(const char *Str) {
  configure(Str, nullptr, 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void UserNamespaceLookalike(const char *Str) {
  (void)mylib::strtol(Str, nullptr, 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

struct Parser {
  long strtol(const char *Str, char **End, int Base);

  void parse(const char *Str) {
    (void)strtol(Str, nullptr, 16);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  }
};

void IndirectCall(const char *Str) {
  long (*Fn)(const char *, char **, int) = strtol;

  (void)Fn(Str, nullptr, 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

void NotAnArgument() {
  int Base = 16;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 16 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  (void)Base;
}
