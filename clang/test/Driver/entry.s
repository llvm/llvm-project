// RUN: %clang -### --target=x86_64-linux-gnu -e foo %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-linux-gnu --entry foo %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-linux-gnu --entry=foo %s 2>&1 | FileCheck %s

// CHECK: "-e" "foo"

/// To prevent mistaking -exxx as --entry=xxx, we allow -e xxx but reject -exxx.
/// GCC -export-dynamic is rejected as well.
// RUN: not %clang -### --target=x86_64-linux-gnu -export-dynamic %s 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: error: unknown argument: '-export-dynamic'
