// RUN: %check_clang_tidy -std=c++17 %s bugprone-suspicious-enum-usage %t -- -config="{CheckOptions: {bugprone-suspicious-enum-usage.StrictMode: false}}"

enum Empty {
};

enum A {
  A = 1,
  B = 2,
  C = 4,
  D = 8,
  E = 16,
  F = 32,
  G = 63
};

enum X {
  X = 8,
  Y = 16,
  Z = 4
};

enum {
  P = 2,
  Q = 3,
  R = 4,
  S = 8,
  T = 16
};

enum {
  H,
  I,
  J,
  K,
  L
};

enum Days {
  Monday,
  Tuesday,
  Wednesday,
  Thursday,
  Friday,
  Saturday,
  Sunday
};

Days bestDay() {
  return Friday;
}

int trigger() {
  Empty EmptyVal;
  int emptytest = EmptyVal | B;
  if (bestDay() | A)
    return 1;
  // CHECK-NOTES: :[[@LINE-2]]:17: warning: enum values are from different enum types
  if (I | Y)
    return 1;
  // CHECK-NOTES: :[[@LINE-2]]:9: warning: enum values are from different enum types
}

int dont_trigger() {
  unsigned p;
  p = Q | P;

  if (A + G == E)
    return 1;
  else if ((Q | R) == T)
    return 1;
  else
    int k = T | Q;

  Empty EmptyVal;
  int emptytest = EmptyVal | B;

  int a = 1, b = 5;
  int c = a + b;
  int d = c | H, e = b * a;
  a = B | C;
  b = X | Z;

  if (Tuesday != Monday + 1 ||
      Friday - Thursday != 1 ||
      Sunday + Wednesday == (Sunday | Wednesday))
    return 1;
  if (H + I + L == 42)
    return 1;
  return 42;
}

namespace PR34400 {
enum { E1 = 0 };
enum { E2 = -1 };
enum { l = E1 | E2 };
}
