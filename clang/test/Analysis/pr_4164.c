// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// expected-no-diagnostics

// PR 4164: http://llvm.org/bugs/show_bug.cgi?id=4164
//
// Eventually this should be pulled into misc-ps.m.  This is in a separate test
// file for now to play around with the specific issues for BasicStoreManager
// and StoreManager (i.e., we can make a copy of this file for either
// StoreManager should one start to fail in the near future).
//
// The basic issue is that the VarRegion for 'size' is casted to (char*),
// resulting in an ElementRegion.  'getsockopt' is an unknown function that
// takes a void*, which means the ElementRegion should get stripped off.
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_socklen_t;
typedef __darwin_socklen_t socklen_t;
int getsockopt(int, int, int, void * restrict, socklen_t * restrict);

int test1() {
  int s = -1;
  int size;
  socklen_t size_len = sizeof(size);
  if (getsockopt(s, 0xffff, 0x1001, (char *)&size, &size_len) < 0)
          return -1;

  return size; // no-warning
}

// Similar case: instead of passing a 'void*', we pass 'char*'.  In this
// case we pass an ElementRegion to the invalidation logic.  Since it is
// an ElementRegion that just layers on top of another typed region and the
// ElementRegion itself has elements whose type are integral (essentially raw
// data) we strip off the ElementRegion when doing the invalidation.
int takes_charptr(char* p);
int test2() {
  int size;
  if (takes_charptr((char*)&size))
    return -1;
  return size; // no-warning
}

