// UNSUPPORTED: windows-msvc

// RUN: %clangxx_cfi -o %t %s
// RUN: %expect_crash %run %t a
// RUN: %expect_crash %run %t b
// RUN: %expect_crash %run %t c
// RUN: %expect_crash %run %t d
// RUN: %expect_crash %run %t e
// RUN: %run %t f
// RUN: %run %t g

// RUN: %clangxx_cfi_diag -o %t2 %s
// RUN: %run %t2 a 2>&1 | FileCheck --check-prefix=A %s
// RUN: %run %t2 b 2>&1 | FileCheck --check-prefix=B %s
// RUN: %run %t2 c 2>&1 | FileCheck --check-prefix=C %s
// RUN: %run %t2 d 2>&1 | FileCheck --check-prefix=D %s
// RUN: %run %t2 e 2>&1 | FileCheck --check-prefix=E %s

#include <assert.h>
#include <string.h>

struct SBase1 {
  void b1() {}
};

struct SBase2 {
  void b2() {}
};

struct S : SBase1, SBase2 {
  void f1() {}
  int f2() { return 1; }
  virtual void g1() {}
  virtual int g2() { return 1; }
  virtual int g3() { return 1; }
};

struct T {
  void f1() {}
  int f2() { return 2; }
  virtual void g1() {}
  virtual int g2() { return 2; }
  virtual void g3() {}
};

typedef void (S::*S_void)();

typedef int (S::*S_int)();
typedef int (T::*T_int)();

template <typename To, typename From>
To bitcast(From f) {
  assert(sizeof(To) == sizeof(From));
  To t;
  memcpy(&t, &f, sizeof(f));
  return t;
}

int main(int argc, char **argv) {
  S s;
  T t;

  switch (argv[1][0]) {
    case 'a':
      // A: runtime error: control flow integrity check for type 'int (S::*)()' failed during non-virtual pointer to member function call
      // A: note: S::f1() defined here
      (s.*bitcast<S_int>(&S::f1))();
      break;
    case 'b':
      // B: runtime error: control flow integrity check for type 'int (T::*)()' failed during non-virtual pointer to member function call
      // B: note: S::f2() defined here
      (t.*bitcast<T_int>(&S::f2))();
      break;
    case 'c':
      // C: runtime error: control flow integrity check for type 'int (S::*)()' failed during virtual pointer to member function call
      // C: note: vtable is of type 'S'
      (s.*bitcast<S_int>(&S::g1))();
      break;
    case 'd':
      // D: runtime error: control flow integrity check for type 'int (S::*)()' failed during virtual pointer to member function call
      // D: note: vtable is of type 'T'
      (reinterpret_cast<S &>(t).*&S::g2)();
      break;
    case 'e':
      // E: runtime error: control flow integrity check for type 'void (S::*)()' failed during virtual pointer to member function call
      // E: note: vtable is of type 'S'
      (s.*bitcast<S_void>(&T::g3))();
      break;
    case 'f':
      (s.*&SBase1::b1)();
      break;
    case 'g':
      (s.*&SBase2::b2)();
      break;
  }
}
