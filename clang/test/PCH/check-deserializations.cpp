// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_method -include-pch %t -emit-llvm-only %s 

#ifndef HEADER
#define HEADER
// Header.

struct S1 {
  void S1_method(); // This should not be deserialized.
  virtual void S1_keyfunc();
};


#else
// Using the header.

void test(S1*) {
}

#endif
