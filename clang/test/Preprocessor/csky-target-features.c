// RUN: %clang -target csky-unknown-linux-gnu  -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK: __CK810__ 1
// CHECK: __CKCORE__ 2
// CHECK: __CSKYABI__ 2
// CHECK: __CSKYLE__ 1
// CHECK: __CSKY__ 2

// CHECK: __ck810__ 1
// CHECK: __ckcore__ 2
// CHECK: __cskyLE__ 1
// CHECK: __csky__ 2
// CHECK: __cskyabi__ 2
// CHECK: __cskyle__ 1
