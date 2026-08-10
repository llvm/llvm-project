// Check that all -fplugin arguments are converted to -load

// RUN: %clang -c %s -fplugin=foo.so -### 2>&1                 | FileCheck %s --check-prefix=CHECK1
// RUN: %clang -c %s -fplugin=foo.so -fplugin=bar.so -### 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK1: "-load" "foo.so"
// CHECK2: "-load" "foo.so" "-load" "bar.so"
