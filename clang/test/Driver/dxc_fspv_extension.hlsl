// If there is no `-fspv-extension`, enable all extensions.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=ALL
// ALL: "-spirv-ext=all"

// Convert the `-fspv-extension` into `spirv-ext`.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=SPV_TEST1 2>&1 | FileCheck %s -check-prefix=TEST1
// TEST1: "-spirv-ext=+SPV_TEST1"

// Merge multiple extensions into a single `spirv-ext` option.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=SPV_TEST1 -fspv-extension=SPV_TEST2 2>&1 | FileCheck %s -check-prefix=TEST2
// TEST2: "-spirv-ext=+SPV_TEST1,+SPV_TEST2"

// Check for the error message if the extension name is not properly formed.
// RUN: not %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=TEST1 -fspv-extension=SPV_GOOD -fspv-extension=TEST2 2>&1 | FileCheck %s -check-prefix=FAIL
// FAIL: invalid value 'TEST1' in '-fspv_extension'
// FAIL: invalid value 'TEST2' in '-fspv_extension'

// If targeting DXIL, the `-spirv-ext` should not be passed to the backend.
// RUN: %clang_dxc -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=DXIL
// DXIL-NOT: spirv-ext

// If targeting DXIL, the `-fspv-extension` option is meaningless, and there should be a warning.
// RUN: %clang_dxc -Tlib_6_7 -### %s -fspv-extension=SPV_TEST 2>&1 | FileCheck %s -check-prefix=WARN
// WARN: warning: argument unused during compilation: '-fspv-extension=SPV_TEST'
// WARN-NOT: spirv-ext
