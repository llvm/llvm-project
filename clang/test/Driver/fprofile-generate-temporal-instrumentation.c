// RUN: %clang -### -c -fprofile-generate -fprofile-generate-temporal-instrumentation %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -fcs-profile-generate -fprofile-generate-temporal-instrumentation %s 2>&1 | FileCheck %s
// RUN: not %clang -### -c -fprofile-generate-temporal-instrumentation %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: "-mllvm" "--pgo-temporal-instrumentation"

// ERR: '-fprofile-generate-temporal-instrumentation' only allowed with '-fprofile-generate or -fcs-profile-generate'
