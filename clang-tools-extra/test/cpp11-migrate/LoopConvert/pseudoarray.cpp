// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
#include "structures.h"

const int N = 6;
dependent<int> v;
dependent<int> *pv;

transparent<dependent<int> > cv;

void f() {
  int sum = 0;
  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v[i]);
    sum += v[i] + 2;
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v.at(i));
    sum += v.at(i) + 2;
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0, e = pv->size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", pv->at(i));
    sum += pv->at(i) + 2;
  }
  // CHECK: for (auto & elem : *pv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  // This test will fail if size() isn't called repeatedly, since it
  // returns unsigned int, and 0 is deduced to be signed int.
  // FIXME: Insert the necessary explicit conversion, or write out the types
  // explicitly.
  for (int i = 0; i < pv->size(); ++i) {
    printf("Fibonacci number is %d\n", (*pv).at(i));
    sum += (*pv)[i] + 2;
  }
  // CHECK: for (auto & elem : *pv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0; i < cv->size(); ++i) {
    printf("Fibonacci number is %d\n", cv->at(i));
    sum += cv->at(i) + 2;
  }
  // CHECK: for (auto & elem : *cv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;
}

// Check for loops that don't mention containers
void noContainer() {
  for (auto i = 0; i < v.size(); ++i) { }
  // CHECK: for (auto & elem : v) { }

  for (auto i = 0; i < v.size(); ++i) ;
  // CHECK: for (auto & elem : v) ;
}
