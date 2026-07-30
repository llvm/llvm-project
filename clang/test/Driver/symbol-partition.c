// RUN: %clang -### -target x86_64-unknown-linux -c -fsymbol-partition=foo %s 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64-pc-win32 -c -fsymbol-partition=foo %s 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: "-fsymbol-partition=foo"
// ERROR: error: unsupported option '-fsymbol-partition=foo' for target 'x86_64-pc-windows-msvc'
