// RUN: rm -rf %t
// RUN: not %clang -target x86_64-unknown-unknown -extract-api --extract-api-ignores=does-not-exist %s 2>&1 | FileCheck %s

// CHECK: fatal error: file 'does-not-exist' specified by '--extract-api-ignores=' not found

void dummy_function(void);
