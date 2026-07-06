/// Tests UseMicrosoftManglingForC inheritance from the Windows host target.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=WIN %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=LINUX %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=LINUX %s

[[clang::sycl_external]] int square(int x) { return x * x; }
// WIN: define {{.*}} @_Z6squarei
// LINUX: define {{.*}} @_Z6squarei

extern "C" [[clang::sycl_external]] int cfunc(int x) { return x + 1; }
// WIN: define {{.*}} @cfunc
// LINUX: define {{.*}} @cfunc
