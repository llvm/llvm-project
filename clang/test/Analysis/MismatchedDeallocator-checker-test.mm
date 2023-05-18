// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.MismatchedDeallocator -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.MismatchedDeallocator -fblocks -DTEST_INLINABLE_ALLOCATORS -verify %s

#include "Inputs/system-header-simulator-objc.h"
#include "Inputs/system-header-simulator-cxx.h"

typedef __typeof__(sizeof(int)) size_t;
void *malloc(size_t);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);
void __attribute((ownership_returns(malloc))) *my_malloc(size_t);

void free(void *);
void __attribute((ownership_takes(malloc, 1))) my_free(void *);

//---------------------------------------------------------------
// Test if an allocation function matches deallocation function
//---------------------------------------------------------------

//--------------- test malloc family
void testMalloc1() {
  int *p = (int *)malloc(sizeof(int));
  delete p; // expected-warning{{Memory allocated by malloc() should be deallocated by free(), not 'delete'}}
}

void testMalloc2() {
  int *p = (int *)malloc(8);
  int *q = (int *)realloc(p, 16);
  delete q; // expected-warning{{Memory allocated by realloc() should be deallocated by free(), not 'delete'}}
}

void testMalloc3() {
  int *p = (int *)calloc(1, sizeof(int));
  delete p; // expected-warning{{Memory allocated by calloc() should be deallocated by free(), not 'delete'}}
}

void testMalloc4(const char *s) {
  char *p = strdup(s);
  delete p; // expected-warning{{Memory allocated by strdup() should be deallocated by free(), not 'delete'}}
}

void testMalloc5() {
  int *p = (int *)my_malloc(sizeof(int));
  delete p; // expected-warning{{Memory allocated by my_malloc() should be deallocated by free(), not 'delete'}}
}

void testMalloc6() {
  int *p = (int *)malloc(sizeof(int));
  operator delete(p); // expected-warning{{Memory allocated by malloc() should be deallocated by free(), not operator delete}}
}

void testMalloc7() {
  int *p = (int *)malloc(sizeof(int));
  delete[] p; // expected-warning{{Memory allocated by malloc() should be deallocated by free(), not 'delete[]'}}
}

void testMalloc8() {
  int *p = (int *)malloc(sizeof(int));
  operator delete[](p); // expected-warning{{Memory allocated by malloc() should be deallocated by free(), not operator delete[]}}
}

void testAlloca() {
  int *p = (int *)__builtin_alloca(sizeof(int));
  delete p; // expected-warning{{Memory allocated by alloca() should not be deallocated}}
}

//--------------- test new family
void testNew1() {
  int *p = new int;
  free(p); // expected-warning{{Memory allocated by 'new' should be deallocated by 'delete', not free()}}
}

void testNew2() {
  int *p = (int *)operator new(0);
  free(p); // expected-warning{{Memory allocated by operator new should be deallocated by 'delete', not free()}}
}

void testNew3() {
  int *p = new int[1];
  free(p); // expected-warning{{Memory allocated by 'new[]' should be deallocated by 'delete[]', not free()}}
}

void testNew4() {
  int *p = new int;
  realloc(p, sizeof(long)); // expected-warning{{Memory allocated by 'new' should be deallocated by 'delete', not realloc()}}
}

void testNew5() {
  int *p = (int *)operator new(0);
  realloc(p, sizeof(long)); // expected-warning{{Memory allocated by operator new should be deallocated by 'delete', not realloc()}}
}

void testNew6() {
  int *p = new int[1];
  realloc(p, sizeof(long)); // expected-warning{{Memory allocated by 'new[]' should be deallocated by 'delete[]', not realloc()}}
}

int *allocInt() {
  return new int;
}
void testNew7() {
  int *p = allocInt();
  delete[] p; // expected-warning{{Memory allocated by 'new' should be deallocated by 'delete', not 'delete[]'}}
}

void testNew8() {
  int *p = (int *)operator new(0);
  delete[] p; // expected-warning{{Memory allocated by operator new should be deallocated by 'delete', not 'delete[]'}}
}

int *allocIntArray(unsigned c) {
  return new int[c];
}

void testNew9() {
  int *p = allocIntArray(1);
  delete p; // expected-warning{{Memory allocated by 'new[]' should be deallocated by 'delete[]', not 'delete'}}
}

void testNew10() {
  int *p = (int *)operator new[](0);
  delete p; // expected-warning{{Memory allocated by operator new[] should be deallocated by 'delete[]', not 'delete'}}
}

void testNew11(NSUInteger dataLength) {
  int *p = new int;
  NSData *d = [NSData dataWithBytesNoCopy:p length:sizeof(int) freeWhenDone:1]; // expected-warning{{+dataWithBytesNoCopy:length:freeWhenDone: cannot take ownership of memory allocated by 'new'}}
}

//-------------------------------------------------------
// Check for intersection with unix.Malloc bounded with 
// unix.MismatchedDeallocator
//-------------------------------------------------------

// new/delete oparators are subjects of cplusplus.NewDelete.
void testNewDeleteNoWarn() {
  int i;
  delete &i; // no-warning

  int *p1 = new int;
  delete ++p1; // no-warning

  int *p2 = new int;
  delete p2;
  delete p2; // no-warning

  int *p3 = new int; // no-warning
}

void testDeleteOpAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  operator delete(p); // no-warning
}

void testDeleteAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  delete p; // no-warning
}

void testStandardPlacementNewAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  p = new(p) int; // no-warning
}

//---------------------------------------------------------------
// Check for intersection with cplusplus.NewDelete bounded with 
// unix.MismatchedDeallocator
//---------------------------------------------------------------

// malloc()/free() are subjects of unix.Malloc and unix.MallocWithAnnotations
void testMallocFreeNoWarn() {
  int i;
  free(&i); // no-warning

  int *p1 = (int *)malloc(sizeof(int));
  free(++p1); // no-warning

  int *p2 = (int *)malloc(sizeof(int));
  free(p2);
  free(p2); // no-warning

  int *p3 = (int *)malloc(sizeof(int)); // no-warning
}

void testFreeAfterDelete() {
  int *p = new int;  
  delete p;
  free(p); // no-warning
}

void testStandardPlacementNewAfterDelete() {
  int *p = new int;  
  delete p;
  p = new(p) int; // no-warning
}


// Smart pointer example
template <typename T>
struct SimpleSmartPointer {
  T *ptr;

  explicit SimpleSmartPointer(T *p = 0) : ptr(p) {}
  ~SimpleSmartPointer() {
    delete ptr;
    // expected-warning@-1 {{Memory allocated by 'new[]' should be deallocated by 'delete[]', not 'delete'}}
    // expected-warning@-2 {{Memory allocated by malloc() should be deallocated by free(), not 'delete'}}
  }
};

void testSimpleSmartPointerArrayNew() {
  {
    SimpleSmartPointer<int> a(new int);
  } // no-warning

  {
    SimpleSmartPointer<int> a(new int[4]);
  }
}

void testSimpleSmartPointerMalloc() {
  {
    SimpleSmartPointer<int> a(new int);
  } // no-warning

  {
    SimpleSmartPointer<int> a((int *)malloc(4));
  }
}
