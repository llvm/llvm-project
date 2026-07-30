// C++ for OpenCL specific logic.
void test(int *ptr) {
  addrspace_cast<__global int*>(ptr);
}

// RUN: c-index-test -test-load-source all %s -cl-std=clc++ -target spir | FileCheck %s
// CHECK: cxx.cl:3:3: CXXAddrspaceCastExpr
