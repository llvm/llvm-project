// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunused-lambda-capture -Wno-unused-value -std=c++1z -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s

void test() {
  int i = 0;
  int j = 0;
  int k = 0;
  int c = 10;
  int a[c];

  [i,j] { return i; };
  // CHECK: [i] { return i; };
  [i,j] { return j; };
  // CHECK: [j] { return j; };
  [&i,j] { return j; };
  // CHECK: [j] { return j; };
  [j,&i] { return j; };
  // CHECK: [j] { return j; };
  [i,j,k] {};
  // CHECK: [] {};
  [i,j,k] { return i + j; };
  // CHECK: [i,j] { return i + j; };
  [i,j,k] { return j + k; };
  // CHECK: [j,k] { return j + k; };
  [i,j,k] { return i + k; };
  // CHECK: [i,k] { return i + k; };
  [i,j,k] { return i + j + k; };
  // CHECK: [i,j,k] { return i + j + k; };
  [&,i] { return k; };
  // CHECK: [&] { return k; };
  [=,&i] { return k; };
  // CHECK: [=] { return k; };
  [=,&i,&j] { return j; };
  // CHECK: [=,&j] { return j; };
  [=,&i,&j] { return i; };
  // CHECK: [=,&i] { return i; };
  [z = i] {};
  // CHECK: [] {};
  [i,z = i] { return z; };
  // CHECK: [z = i] { return z; };
  [z = i,i] { return z; };
  // CHECK: [z = i] { return z; };
  [&a] {};
  // CHECK: [] {};
  [i,&a] { return i; };
  // CHECK: [i] { return i; };
  [&a,i] { return i; };
  // CHECK: [i] { return i; };

  #define I_MACRO() i
  #define I_REF_MACRO() &i
  [I_MACRO()] {};
  // CHECK: [] {};
  [I_MACRO(),j] { return j; };
  // CHECK: [j] { return j; };
  [j,I_MACRO()] { return j; };
  // CHECK: [j] { return j; };
  [I_REF_MACRO(),j] { return j; };
  // CHECK: [j] { return j; };
  [j,I_REF_MACRO()] { return j; };
  // CHECK: [j] { return j; };

  int n = 0;
  [z = (n = i),j] {};
  // CHECK: [z = (n = i)] {};
  [j,z = (n = i)] {};
  // CHECK: [z = (n = i)] {};
}

class ThisTest {
  void test() {
    int i = 0;

    [this] {};
    // CHECK: [] {};
    [i,this] { return i; };
    // CHECK: [i] { return i; };
    [this,i] { return i; };
    // CHECK: [i] { return i; };
    [*this] {};
    // CHECK: [] {};
    [*this,i] { return i; };
    // CHECK: [i] { return i; };
    [i,*this] { return i; };
    // CHECK: [i] { return i; };
    [*this] { return this; };
    // CHECK: [*this] { return this; };
    [*this,i] { return this; };
    // CHECK: [*this] { return this; };
    [i,*this] { return this; };
    // CHECK: [*this] { return this; };
  }
};
