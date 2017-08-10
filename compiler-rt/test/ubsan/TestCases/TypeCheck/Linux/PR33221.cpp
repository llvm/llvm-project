// RUN: %clangxx -std=c++11 -frtti -fsanitize=vptr -g %s -O3 -o %t
// RUN: %run %t &> %t.log
// RUN: cat %t.log | not count 0 && FileCheck --input-file %t.log %s || cat %t.log | count 0

// REQUIRES: cxxabi

#include <sys/mman.h>
#include <unistd.h>

class Base {
public:
  int i;
  virtual void print() {}
};

class Derived : public Base {
public:
  void print() {}
};


int main() {
  int page_size = getpagesize();

  void *non_accessible = mmap(nullptr, page_size, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (non_accessible == MAP_FAILED)
    return 0;

  void *accessible = mmap((char*)non_accessible + page_size, page_size,
                          PROT_READ | PROT_WRITE,
                          MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (accessible == MAP_FAILED)
    return 0;

  char *c = new char[sizeof(Derived)];

  // The goal is to trigger a condition when Vptr points to accessible memory,
  // but VptrPrefix does not. That has been triggering SIGSEGV in UBSan code.
  void **vtable_ptr = reinterpret_cast<void **>(c);
  *vtable_ptr = (void*)accessible;

  Derived *list = (Derived *)c;

// CHECK: PR33221.cpp:[[@LINE+2]]:19: runtime error: member access within address {{.*}} which does not point to an object of type 'Base'
// CHECK-NEXT: invalid vptr
  int foo = list->i;
  return 0;
}
