// Checks that cuda compilation does the right thing when passed -fcuda-short-ptr

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -fcuda-short-ptr -nocudainc -nocudalib %s 2>&1 | FileCheck %s

// CHECK-NOT: "--nvptx-short-ptr"
// CHECK: "-target-abi" "shortptr"
// CHECK-NOT: "-fcuda-short-ptr"
