// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-linux-gnu -analyzer-checker=optin.performance -analyzer-config optin.performance.Padding:AllowedPad=2 -verify %s

class Empty {}; // no-warning

// expected-warning@+1{{Excessive padding in 'struct NoUniqueAddressWarn1' (6 padding}}
struct NoUniqueAddressWarn1 {
  char c1;
  [[no_unique_address]] Empty empty;
  int i;
  char c2;
};

// expected-warning@+1{{Excessive padding in 'struct NoUniqueAddressWarn2' (6 padding}}
struct NoUniqueAddressWarn2 {
    char c1;
    [[no_unique_address]] Empty e1, e2;
    int i;
    char c2;
};

struct NoUniqueAddressNoWarn1 {
  char c1;
  [[no_unique_address]] Empty empty;
  char c2;
};

struct NoUniqueAddressNoWarn2 {
  char c1;
  [[no_unique_address]] Empty e1, e2;
};
