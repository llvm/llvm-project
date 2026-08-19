// RUN: %clang_cc1 -triple arm64e-apple-ios -fno-plt -emit-llvm %s -o - | FileCheck %s --check-prefix=ARM64E
// RUN: %clang_cc1 -triple arm64-apple-ios -fno-plt -emit-llvm %s -o - | FileCheck %s --check-prefix=ARM64
//
// On arm64e, NonLazyBind must NOT be applied. NonLazyBind causes inline
// GOT loads that bypass the linker's authentication stubs. Calls must go
// through stubs that load from __auth_got and authenticate via braa.

void external_function(void);

void caller(void) {
  external_function();
}

// arm64e: the declaration should NOT have nonlazybind.
// ARM64E: declare{{.*}} void @external_function()
// ARM64E-NOT: nonlazybind

// arm64: -fno-plt adds nonlazybind normally.
// ARM64: ; Function Attrs:{{.*}}nonlazybind
// ARM64: declare{{.*}} void @external_function()
