// If there is no `-fspv-extension`, enable the extensions DXC implements, and
// not SPV_KHR_fma, SPV_KHR_float_controls2 or SPV_EXT_long_vector.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=DEFAULT --implicit-check-not=SPV_KHR_fma --implicit-check-not=SPV_KHR_float_controls2 --implicit-check-not=SPV_EXT_long_vector --implicit-check-not=-spirv-ext=all
// DEFAULT: "-spirv-ext={{(\+SPV_[a-zA-Z0-9_]+,?)+}}"

// `-fspv-extension=all` enables every extension the backend supports.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=all 2>&1 | FileCheck %s -check-prefix=ALL
// ALL: "-spirv-ext=all"

// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=all -fspv-extension=SPV_TEST1 2>&1 | FileCheck %s -check-prefix=ALL2
// ALL2: "-spirv-ext=all,+SPV_TEST1"

// Convert the `-fspv-extension` into `spirv-ext`.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=SPV_TEST1 2>&1 | FileCheck %s -check-prefix=TEST1
// TEST1: "-spirv-ext=+SPV_TEST1"

// Merge multiple extensions into a single `spirv-ext` option.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=SPV_TEST1 -fspv-extension=SPV_TEST2 2>&1 | FileCheck %s -check-prefix=TEST2
// TEST2: "-spirv-ext=+SPV_TEST1,+SPV_TEST2"

// Merge KHR with other extensions.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=SPV_TEST1 -fspv-extension=KHR -fspv-extension=SPV_TEST2 2>&1 | FileCheck %s -check-prefix=TEST3
// TEST3: "-spirv-ext=+SPV_TEST1,KHR,+SPV_TEST2"

// Merge KHR with other extensions.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=KHR -fspv-extension=SPV_TEST2 2>&1 | FileCheck %s -check-prefix=TEST4
// TEST4: "-spirv-ext=KHR,+SPV_TEST2"

// Convert DXC to a list of SPV extensions.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=DXC 2>&1 | FileCheck %s -check-prefix=TEST5
// TEST5: "-spirv-ext={{(\+SPV_[a-zA-Z0-9_]+,?)+}}"

// Check for the error message if the extension name is not properly formed.
// RUN: not %clang_dxc -spirv -Tlib_6_7 -### %s -fspv-extension=KHR_BAD -fspv-extension=TEST1 -fspv-extension=SPV_GOOD -fspv-extension=TEST2 2>&1 | FileCheck %s -check-prefix=FAIL
// FAIL: invalid value 'KHR_BAD' in '-fspv-extension'
// FAIL: invalid value 'TEST1' in '-fspv-extension'
// FAIL: invalid value 'TEST2' in '-fspv-extension'

// If targeting DXIL, the `-spirv-ext` should not be passed to the backend.
// RUN: %clang_dxc -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=DXIL
// DXIL-NOT: spirv-ext

// If targeting DXIL, the `-fspv-extension` option is meaningless, and there should be a warning.
// RUN: %clang_dxc -Tlib_6_7 -### %s -fspv-extension=SPV_TEST 2>&1 | FileCheck %s -check-prefix=WARN
// WARN: warning: argument unused during compilation: '-fspv-extension=SPV_TEST'
// WARN-NOT: spirv-ext
