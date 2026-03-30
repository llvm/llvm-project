// RUN: %clang --analyze %s 2>&1 | FileCheck %s

// CHECK: Address of stack memory associated with local variable 'i'
// CHECK: is still referred to by the global variable 'gp' upon returning
// CHECK: to the caller. This will be a dangling reference
// CHECK: Potential leak of memory pointed to by 'p'

unsigned int *gp;
int foo(unsigned int argc) {
  int *p = new int[10]{};
  unsigned int i = 100;
  gp = &i;
  if (argc > *p)
    return i;
  return *p;
}
