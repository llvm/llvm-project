// Make sure we can process CUDA file even if driver creates multiple jobs
// RUN: c-index-test -test-load-source all %s | FileCheck %s -check-prefix=CHECK-ANY
// Make sure we process correct side of cuda compilation
// RUN: c-index-test -test-load-source all --cuda-host-only %s | FileCheck %s -check-prefix=CHECK-HOST
// RUN: c-index-test -test-load-source all --cuda-device-only %s | FileCheck %s -check-prefix=CHECK-DEVICE

// CHECK-ANY: macro definition=__cplusplus
// CHECK-HOST-NOT: macro definition=__CUDA_ARCH__
// CHECK-DEVICE: macro definition=__CUDA_ARCH__
