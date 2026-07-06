/// Tests that SPIRV64 compiles with different getBuiltinVaListKind() delegations.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=LINUX %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=WIN %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=NOHOST %s

// LINUX: #define __SPIRV64__ 1
// WIN: #define __SPIRV64__ 1
// NOHOST: #define __SPIRV64__ 1
