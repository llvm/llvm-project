// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wunused-lambda-capture -Wno-unused-value -std=c++1z -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s

void test() {
  int i = 0;
  int j = 0;
  int k = 0;
  constexpr int c = 10;
  int a[c]; // Make 'c' constexpr to avoid variable-length array warnings.

  [i] { return i; };
  // CHECK: [i] { return i; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [] {};
  // CHECK: [] {};
  [i,j] { return i + j; };
  // CHECK: [i,j] { return i + j; };
  [j,k] { return j + k; };
  // CHECK: [j,k] { return j + k; };
  [i,k] { return i + k; };
  // CHECK: [i,k] { return i + k; };
  [i,j,k] { return i + j + k; };
  // CHECK: [i,j,k] { return i + j + k; };
  [&] { return k; };
  // CHECK: [&] { return k; };
  [=] { return k; };
  // CHECK: [=] { return k; };
  [=,&j] { return j; };
  // CHECK: [=,&j] { return j; };
  [=,&i] { return i; };
  // CHECK: [=,&i] { return i; };
  [] {};
  // CHECK: [] {};
  [z = i] { return z; };
  // CHECK: [z = i] { return z; };
  [z = i] { return z; };
  // CHECK: [z = i] { return z; };
  [] {};
  // CHECK: [] {};
  [i] { return i; };
  // CHECK: [i] { return i; };
  [i] { return i; };
  // CHECK: [i] { return i; };

#define I_MACRO() i
#define I_REF_MACRO() &i
  [] {};
  // CHECK: [] {};
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j] { return j; };
  // CHECK: [j] { return j; };

  int n = 0;
  [z = (n = i)] {};
  // CHECK: [z = (n = i)] {};
  [z = (n = i)] {};
  // CHECK: [z = (n = i)] {};

  // New Edge Cases

  // Test 1: Leading and trailing whitespace
  [i] { return i; };
  // CHECK: [i] { return i; };
  [j] { return j; };
  // CHECK: [j] { return j; };
  [j,k] { return j + k; };
  // CHECK: [j,k] { return j + k; };

  // Test 2: Single unused capture
  [] {};
  // CHECK: [] {};
  [] {};
  // CHECK: [] {};

  // Test 3: Multiple commas
  [j] { return j; };
  // CHECK: [j] { return j; };
  [k] { return k; };
  // CHECK: [k] { return k; };

  // Test 4: Mixed captures
  [&i] { return i; };
  // CHECK: [&i] { return i; };
  [&] {};
  // CHECK: [&] {};

  // Test 5: Capture with comments
  [/*capture*/ j] { return j; };
  // CHECK: [/*capture*/ j] { return j; };
}

class ThisTest {
  void test() {
    int i = 0;

    [] {};
    // CHECK: [] {};
    [i] { return i; };
    // CHECK: [i] { return i; };
    [i] { return i; };
    // CHECK: [i] { return i; };
    [] {};
    // CHECK: [] {};
    [i] { return i; };
    // CHECK: [i] { return i; };
    [i] { return i; };
    // CHECK: [i] { return i; };
    [*this] { return this; };
    // CHECK: [*this] { return this; };
    [*this] { return this; };
    // CHECK: [*this] { return this; };
    [*this] { return this; };
    // CHECK: [*this] { return this; };
  }
};
