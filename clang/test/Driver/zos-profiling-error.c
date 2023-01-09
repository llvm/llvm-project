// RUN: %clang 2>&1 -### --target=s390x-none-zos -pg -S %s | FileCheck -check-prefix=FAIL-PG-NAME %s
// FAIL-PG-NAME: error: unsupported option '-pg' for target 's390x-none-zos'
