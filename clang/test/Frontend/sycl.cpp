// Test that we disallow -cc1 -fsycl, even when specifying device or host mode.

// RUN: not %clang_cc1 -fsycl %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang_cc1 -fsycl -fsycl-is-device %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang_cc1 -fsycl -fsycl-is-host %s 2>&1 | FileCheck --check-prefix=ERROR %s

// ERROR: error: unknown argument: '-fsycl'

// Test that you cannot specify -fsycl-is-device and -fsycl-is-host at the same time.
// RUN: not %clang_cc1 -fsycl-is-device -fsycl-is-host %s 2>&1 | FileCheck --check-prefix=ERROR-BOTH %s
// RUN: not %clang_cc1 -fsycl-is-host -fsycl-is-device %s 2>&1 | FileCheck --check-prefix=ERROR-BOTH %s

// ERROR-BOTH: error: invalid argument '-fsycl-is-device' not allowed with '-fsycl-is-host'
