// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,unix.Malloc -std=c++11 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,unix.Malloc -std=c++11 -DLEAKS=1 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,unix.Malloc -std=c++11 -analyzer-config c++-allocator-inlining=true -DALLOCATOR_INLINING=1 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks,unix.Malloc -std=c++11 -analyzer-config c++-allocator-inlining=true -DLEAKS=1 -DALLOCATOR_INLINING=1 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"

#if !(LEAKS && !ALLOCATOR_INLINING)
// expected-no-diagnostics
#endif


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

  C *p2 = new C; // no warn

  C *c3 = ::new C;
}
#if LEAKS && !ALLOCATOR_INLINING
// expected-warning@-2{{Potential leak of memory pointed to by 'c3'}}
#endif

void testOpNewArray() {
  void *p = operator new[](0); // call is inlined, no warn
}

void testNewExprArray() {
  int *p = new int[0];
}
#if LEAKS && !ALLOCATOR_INLINING
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif


//----- Custom non-placement operators
void testOpNew() {
  void *p = operator new(0); // call is inlined, no warn
}

void testNewExpr() {
  int *p = new int;
}
#if LEAKS && !ALLOCATOR_INLINING
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif


//----- Custom NoThrow placement operators
void testOpNewNoThrow() {
  void *p = operator new(0, std::nothrow); // call is inlined, no warn
}

void testNewExprNoThrow() {
  int *p = new(std::nothrow) int;
}
#if LEAKS && !ALLOCATOR_INLINING
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif

//----- Custom placement operators
void testOpNewPlacement() {
  void *p = operator new(0, 0.1); // no warn
}

void testNewExprPlacement() {
  int *p = new(0.1) int; // no warn
}
