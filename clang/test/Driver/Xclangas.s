/// Check that -Xclangas/-Xclangas= are passed to -cc1as.
// RUN: %clang -### -Werror -Xclangas -target-feature -Xclangas=+v5t %s 2>&1 | FileCheck %s
// CHECK: -cc1as
// CHECK-SAME: "-target-feature" "+v5t"
// XFAIL: target={{.*}}-aix{{.*}}
