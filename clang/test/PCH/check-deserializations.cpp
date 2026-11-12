// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-pch -o %t.1 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -error-on-deserialized-decl S1_keyfunc -error-on-deserialized-decl S3 -error-on-deserialized-decl DND -std=c++11 -include-pch %t.1 -emit-pch -o %t.2 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -error-on-deserialized-decl S1_method -error-on-deserialized-decl S3 -error-on-deserialized-decl DND -std=c++11 -include-pch %t.2 -emit-llvm-only %s

// FIXME: Why does this require an x86 target?
// REQUIRES: x86-registered-target

#ifndef HEADER1
#define HEADER1
// Header.

struct S1 {
  void S1_method();
  virtual void S1_keyfunc();
};

struct S3 {};

struct S2 {
  operator S3();
};

namespace vars {
  constexpr int f() { return 0; }
  struct X { constexpr X() {} };
  namespace v1 { const int DND = 0; }
  namespace v2 { constexpr int DND = f(); }
  namespace v3 { static X DND; }
  namespace v4 { constexpr X DND = {}; }
}

#elif !defined(HEADER2)
#define HEADER2

// Chained PCH.
S1 *s1;
S2 *s2;

#else

// Using the headers.

void test(S1*, S2*) {
}

#endif
