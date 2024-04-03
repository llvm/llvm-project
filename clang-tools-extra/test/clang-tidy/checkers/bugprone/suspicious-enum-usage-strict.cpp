// RUN: %check_clang_tidy -std=c++17 %s bugprone-suspicious-enum-usage %t -- -config="{CheckOptions: {bugprone-suspicious-enum-usage.StrictMode: true}}" --

enum A {
  A = 1,
  B = 2,
  C = 4,
  D = 8,
  E = 16,
  F = 32,
  G = 63
};

// CHECK-NOTES: :[[@LINE+2]]:1: warning: enum type seems like a bitmask (contains mostly power-of-2 literals) but a literal is not power-of-2
// CHECK-NOTES: :76:7: note: used here as a bitmask
enum X {
  X = 8,
  Y = 16,
  Z = 4,
  ZZ = 3
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: enum type seems like a bitmask (contains mostly power-of-2 literals), but this literal is not a power-of-2 [bugprone-suspicious-enum-usage]
// CHECK-NOTES: :70:13: note: used here as a bitmask
};
// CHECK-NOTES: :[[@LINE+2]]:1: warning: enum type seems like a bitmask (contains mostly power-of-2 literals) but some literals are not power-of-2
// CHECK-NOTES: :73:8: note: used here as a bitmask
enum PP {
  P = 2,
  Q = 3,
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: enum type seems like a bitmask (contains mostly power-of-2 literals), but this literal is not a power-of-2
  // CHECK-NOTES: :65:11: note: used here as a bitmask
  R = 4,
  S = 8,
  T = 16,
  U = 31
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
  if (bestDay() | A)
    return 1;
  // CHECK-NOTES: :[[@LINE-2]]:17: warning: enum values are from different enum types
  if (I | Y)
    return 1;
  // CHECK-NOTES: :[[@LINE-2]]:9: warning: enum values are from different enum types
  if (P + Q == R)
    return 1;
  else if ((S | R) == T)
    return 1;
  else
    int k = ZZ | Z;
  unsigned p = R;
  PP pp = Q;
  p |= pp;

  enum X x = Z;
  p = x | Z;
  return 0;
}

int dont_trigger() {
  int a = 1, b = 5;
  int c = a + b;
  int d = c | H, e = b * a;
  a = B | C;
  b = X | Z;

  unsigned bitflag;
  enum A aa = B;
  bitflag = aa | C;

  if (Tuesday != Monday + 1 ||
      Friday - Thursday != 1 ||
      Sunday + Wednesday == (Sunday | Wednesday))
    return 1;
  if (H + I + L == 42)
    return 1;
  return 42;
}
