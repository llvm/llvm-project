// RUN: %clang_cc1 -fsyntax-only -std=c++11 2>&1 %s | FileCheck -strict-whitespace %s

struct A {
  unsigned int a;
};

// PR10696
void testOverlappingInsertions(int b) {
  A var = { b };
  // CHECK:  A var = { b };
  // CHECK:            ^
  // CHECK:            static_cast<unsigned int>( )
}
