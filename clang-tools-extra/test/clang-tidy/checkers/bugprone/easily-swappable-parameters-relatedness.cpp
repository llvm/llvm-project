// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 1, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 0 \
// RUN:  }}' --

namespace std {
template <typename T>
T max(const T &A, const T &B);
} // namespace std

bool coin();
void f(int);
void g(int);
void h(int, int);
void i(int, bool);
void i(int, char);

struct Tmp {
  int f(int);
  int g(int, int);
};

struct Int {
  int I;
};

void compare(int Left, int Right) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 2 adjacent parameters of 'compare' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:18: note: the first parameter in the range is 'Left'
// CHECK-MESSAGES: :[[@LINE-3]]:28: note: the last parameter in the range is 'Right'

int decideSequence(int A, int B) {
  if (A)
    return 1;
  if (B)
    return 2;
  return 3;
}
// CHECK-MESSAGES: :[[@LINE-7]]:20: warning: 2 adjacent parameters of 'decideSequence' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-8]]:24: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-9]]:31: note: the last parameter in the range is 'B'

int myMax(int A, int B) { // NO-WARN: Appears in same expression.
  return A < B ? A : B;
}

int myMax2(int A, int B) { // NO-WARN: Appears in same expression.
  if (A < B)
    return A;
  return B;
}

int myMax3(int A, int B) { // NO-WARN: Appears in same expression.
  return std::max(A, B);
}

int binaryToUnary(int A, int) {
  return A;
}
// CHECK-MESSAGES: :[[@LINE-3]]:19: warning: 2 adjacent parameters of 'binaryToUnary' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-4]]:23: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-5]]:29: note: the last parameter in the range is '<unnamed>'

int randomReturn1(int A, int B) { // NO-WARN: Appears in same expression.
  return coin() ? A : B;
}

int randomReturn2(int A, int B) { // NO-WARN: Both parameters returned.
  if (coin())
    return A;
  return B;
}

int randomReturn3(int A, int B) { // NO-WARN: Both parameters returned.
  bool Flip = coin();
  if (Flip)
    return A;
  Flip = coin();
  if (Flip)
    return B;
  Flip = coin();
  if (!Flip)
    return 0;
  return -1;
}

void passthrough1(int A, int B) { // WARN: Different functions, different params.
  f(A);
  g(B);
}
// CHECK-MESSAGES: :[[@LINE-4]]:19: warning: 2 adjacent parameters of 'passthrough1' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-6]]:30: note: the last parameter in the range is 'B'

void passthrough2(int A, int B) { // NO-WARN: Passed to same index of same function.
  f(A);
  f(B);
}

void passthrough3(int A, int B) { // NO-WARN: Passed to same index of same funtion.
  h(1, A);
  h(1, B);
}

void passthrough4(int A, int B) { // WARN: Different index used.
  h(1, A);
  h(B, 2);
}
// CHECK-MESSAGES: :[[@LINE-4]]:19: warning: 2 adjacent parameters of 'passthrough4' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-6]]:30: note: the last parameter in the range is 'B'

void passthrough5(int A, int B) { // WARN: Different function overload.
  i(A, false);
  i(A, '\0');
}
// CHECK-MESSAGES: :[[@LINE-4]]:19: warning: 2 adjacent parameters of 'passthrough5' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-6]]:30: note: the last parameter in the range is 'B'

void passthrough6(int A, int B) { // NO-WARN: Passed to same index of same function.
  Tmp Temp;
  Temp.f(A);
  Temp.f(B);
}

void passthrough7(int A, int B) { // NO-WARN: Passed to same index of same function.
  // Clang-Tidy isn't path sensitive, the fact that the two objects we call the
  // function on is different is not modelled.
  Tmp Temp1, Temp2;
  Temp1.f(A);
  Temp2.f(B);
}

void passthrough8(int A, int B) { // WARN: Different functions used.
  f(A);
  Tmp{}.f(B);
}
// CHECK-MESSAGES: :[[@LINE-4]]:19: warning: 2 adjacent parameters of 'passthrough8' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-6]]:30: note: the last parameter in the range is 'B'

// Test that the matching of "passed-to-function" is done to the proper node.
// Put simply, this test should not crash here.
void forwardDeclared(int X);

void passthrough9(int A, int B) { // NO-WARN: Passed to same index of same function.
  forwardDeclared(A);
  forwardDeclared(B);
}

void forwardDeclared(int X) {}

void passthrough10(int A, int B) { // NO-WARN: Passed to same index of same function.
  forwardDeclared(A);
  forwardDeclared(B);
}

bool compare1(Int I, Int J) { // NO-WARN: Same member accessed.
  int Val1 = I.I;
  int Val2 = J.I;
  return Val1 < Val2;
}

bool compare2(Tmp T1, Tmp T2) { // NO-WARN: Same member accessed.
  int Val1 = T1.g(0, 1);
  int Val2 = T2.g(2, 3);
  return Val1 < Val2;
}

bool compare3(Tmp T1, Tmp T2) { // WARN: Different member accessed.
  int Val1 = T1.f(0);
  int Val2 = T2.g(1, 2);
  return Val1 < Val2;
}
// CHECK-MESSAGES: :[[@LINE-5]]:15: warning: 2 adjacent parameters of 'compare3' of similar type ('Tmp')
// CHECK-MESSAGES: :[[@LINE-6]]:19: note: the first parameter in the range is 'T1'
// CHECK-MESSAGES: :[[@LINE-7]]:27: note: the last parameter in the range is 'T2'

int rangeBreaker(int I, int J, int K, int L, int M, int N) {
  // (I, J) swappable.

  if (J == K) // (J, K) related.
    return -1;

  if (K + 2 > Tmp{}.f(K))
    return M;

  // (K, L, M) swappable.

  return N; // (M, N) related.
}
// CHECK-MESSAGES: :[[@LINE-13]]:18: warning: 2 adjacent parameters of 'rangeBreaker' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-14]]:22: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-15]]:29: note: the last parameter in the range is 'J'
// CHECK-MESSAGES: :[[@LINE-16]]:32: warning: 3 adjacent parameters of 'rangeBreaker' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-17]]:36: note: the first parameter in the range is 'K'
// CHECK-MESSAGES: :[[@LINE-18]]:50: note: the last parameter in the range is 'M'

int returnsNotOwnParameter(int I, int J, int K) {
  const auto &Lambda = [&K](int L, int M, int N) {
    if (K)
      return L;
    return M; // (L, M) related.
  };

  if (Lambda(-1, 0, 1))
    return I;
  return J; // (I, J) related.
}
// CHECK-MESSAGES: :[[@LINE-11]]:35: warning: 2 adjacent parameters of 'returnsNotOwnParameter' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-12]]:39: note: the first parameter in the range is 'J'
// CHECK-MESSAGES: :[[@LINE-13]]:46: note: the last parameter in the range is 'K'
// CHECK-MESSAGES: :[[@LINE-13]]:36: warning: 2 adjacent parameters of 'operator()' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-14]]:40: note: the first parameter in the range is 'M'
// CHECK-MESSAGES: :[[@LINE-15]]:47: note: the last parameter in the range is 'N'

int usedTogetherInCapture(int I, int J, int K) { // NO-WARN: Used together.
  const auto &Lambda = [I, J, K]() {
    int A = I + 1;
    int B = J - 2;
    int C = K * 3;
    return A + B + C;
  };
  return Lambda();
}
