// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s


// ref-no-diagnostics

constexpr int a = 10;
constexpr const int &b = a;
static_assert(a == b, "");

constexpr int assignToReference() {
  int a = 20;
  int &b = a;

  b = 100;
  return a;
}
static_assert(assignToReference() == 100, "");


constexpr void setValue(int &dest, int val) {
  dest = val;
}

constexpr int checkSetValue() {
  int l = 100;
  setValue(l, 200);
  return l;
}
static_assert(checkSetValue() == 200, "");

constexpr int readLocalRef() {
  int a = 20;
  int &b = a;
  return b;
}
static_assert(readLocalRef() == 20, "");

constexpr int incRef() {
  int a = 0;
  int &b = a;

  b = b + 1;

  return a;
}
static_assert(incRef() == 1, "");


template<const int &V>
constexpr void Plus3(int &A) {
  A = V + 3;
}
constexpr int foo = 4;

constexpr int callTemplate() {
  int a = 3;
  Plus3<foo>(a);
  return a;
}
static_assert(callTemplate() == 7, "");


constexpr int& getValue(int *array, int index) {
  return array[index];
}
constexpr int testGetValue() {
  int values[] = {1, 2, 3, 4};
  getValue(values, 2) = 30;
  return values[2];
}
static_assert(testGetValue() == 30, "");

// FIXME: ExprWithCleanups + MaterializeTemporaryExpr not implemented
constexpr const int &MCE = 1; // expected-error{{must be initialized by a constant expression}}


struct S {
  int i, j;
};

constexpr int RefToMemberExpr() {
  S s{1, 2};

  int &j = s.i;
  j = j + 10;

  return j;
}
static_assert(RefToMemberExpr() == 11, "");

struct Ref {
  int &a;
};

constexpr int RecordWithRef() {
  int m = 100;
  Ref r{m};
  m = 200;
  return r.a;
}
static_assert(RecordWithRef() == 200, "");


struct Ref2 {
  int &a;
  constexpr Ref2(int &a) : a(a) {}
};

constexpr int RecordWithRef2() {
  int m = 100;
  Ref2 r(m);
  m = 200;
  return r.a;
}
static_assert(RecordWithRef2() == 200, "");
