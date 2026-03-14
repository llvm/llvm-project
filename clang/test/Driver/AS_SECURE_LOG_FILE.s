// RUN: env AS_SECURE_LOG_FILE=log_file %clang --target=x86_64-apple-darwin -c %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1as"
// CHECK-SAME: "-as-secure-log-file" "log_file"
