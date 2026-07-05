// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,unix.Malloc -std=c++11 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,unix.Malloc -std=c++11 -fblocks -verify %s -analyzer-config c++-allocator-inlining=false
#include "Inputs/system-header-simulator-cxx.h"

// expected-no-diagnostics


void *allocator(std::size_t size);

void *operator new[](std::size_t size) throw() { return allocator(size); }
void *operator new(std::size_t size) throw() { return allocator(size); }
void *operator new(std::size_t size, const std::nothrow_t &nothrow) throw() { return allocator(size); }
void *operator new(std::size_t, double d);

class C {
public:
  void *operator new(std::size_t);  
};

void testNewMethod() {
  void *p1 = C::operator new(0); // no warn

  C *p2 = new C; // no-warning

  C *c3 = ::new C; // no-warning
}

void testOpNewArray() {
  void *p = operator new[](0); // call is inlined, no warn
}

void testNewExprArray() {
  int *p = new int[0]; // no-warning
}


//----- Custom non-placement operators
void testOpNew() {
  void *p = operator new(0); // call is inlined, no warn
}

void testNewExpr() {
  int *p = new int; // no-warning
}

//----- Custom NoThrow placement operators
void testOpNewNoThrow() {
  void *p = operator new(0, std::nothrow); // call is inlined, no warn
}

void testNewExprNoThrow() {
  int *p = new(std::nothrow) int; // no-warning
}

//----- Custom placement operators
void testOpNewPlacement() {
  void *p = operator new(0, 0.1); // no warn
}

void testNewExprPlacement() {
  int *p = new(0.1) int; // no warn
}
