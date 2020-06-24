// RUN: not %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm < %s \
// RUN:     2>&1 | \
// RUN:   FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm < %s \
// RUN:     2>&1 | \
// RUN:   FileCheck %s

class test {
  int a;

public:
  test(int c) { a = c; }
  ~test() { a = 0; }
};

__attribute__((init_priority(2000)))
test t(1);

// CHECK: fatal error: error in backend: 'init_priority' attribute is not yet supported on AIX
