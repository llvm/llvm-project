// RUN: %clang -E %s -### 2>&1 | FileCheck %s -check-prefix=NO_EPOCH
// NO_EPOCH-NOT: "-source-date-epoch"

// RUN: env SOURCE_DATE_EPOCH=123 %clang -E %s -### 2>&1 | FileCheck %s
// CHECK: "-source-date-epoch" "123"
