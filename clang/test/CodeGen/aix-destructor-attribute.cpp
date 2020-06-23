// RUN: not %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm < %s \
// RUN:     2>&1 | \
// RUN:   FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm < %s \
// RUN:     2>&1 | \
// RUN:   FileCheck %s

int bar() __attribute__((destructor(180)));

class test {
  int a;

public:
  test(int c) { a = c; }
  ~test() { a = 0; }
};

test t(1);

// CHECK: fatal error: error in backend: 'destructor' attribute is not yet supported on AIX
