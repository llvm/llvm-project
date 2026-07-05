// RUN: %check_clang_tidy %s modernize-loop-convert %t

template <class T>
struct Vector {
  using iterator = T*;
  unsigned size() const;
  const T &operator[](int) const;
  T &operator[](int);
  T *begin();
  T *end();
  const T *begin() const;
  const T *end() const;
};

template <typename T>
void copyArg(T);

class TestArrayOfVector {
  Vector<int> W[2];

  void foo() const {
    for (int I = 0; I < W[0].size(); ++I) {
      if (W[0][I])
        copyArg(W[0][I]);
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop
    // CHECK-FIXES: for (int I : W[0]) {
    // CHECK-FIXES-NEXT: if (I)
    // CHECK-FIXES-NEXT: copyArg(I);
  }
};

class TestVectorOfVector {
  Vector<Vector<int>> X;

  void foo() const {
    for (int J = 0; J < X[0].size(); ++J) {
      if (X[0][J])
        copyArg(X[0][J]);
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop
    // CHECK-FIXES: for (int J : X[0]) {
    // CHECK-FIXES-NEXT: if (J)
    // CHECK-FIXES-NEXT: copyArg(J);
  }
};

void testVectorOfVectorOfVector() {
  Vector<Vector<Vector<int>>> Y;
  for (int J = 0; J < Y[3].size(); ++J) {
    if (Y[3][J][7])
      copyArg(Y[3][J][8]);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop
  // CHECK-FIXES: for (auto & J : Y[3]) {
  // CHECK-FIXES-NEXT: if (J[7])
  // CHECK-FIXES-NEXT: copyArg(J[8]);

  for (int J = 0; J < Y[3][4].size(); ++J) {
    if (Y[3][4][J])
      copyArg(Y[3][4][J]);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop
  // CHECK-FIXES: for (int J : Y[3][4]) {
  // CHECK-FIXES-NEXT: if (J)
  // CHECK-FIXES-NEXT: copyArg(J);
}

void testVectorOfVectorIterator() {
  Vector<Vector<int>> Z;
  for (Vector<int>::iterator it = Z[4].begin(); it != Z[4].end();  ++it) {
    if (*it)
      copyArg(*it);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop
  // CHECK-FIXES: for (int & it : Z[4]) {
  // CHECK-FIXES-NEXT: if (it)
  // CHECK-FIXES-NEXT: copyArg(it);
}
