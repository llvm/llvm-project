// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -verify %s
//
// expected-no-diagnostics

typedef long off_t;
typedef long long off64_t;
typedef unsigned long size_t;

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off64_t offset);

void test(long len) {
  mmap(0, len, 2, 1, 0, 0);   // no-crash
  mmap64(0, len, 2, 1, 0, 0); // no-crash
}
