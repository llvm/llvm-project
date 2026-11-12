//
// Verify -fgnuc-version parsing
//

// RUN: %clang -c %s --target=i686-linux -### 2>&1 | FileCheck %s --check-prefix=GNUC-DEFAULT
// GNUC-DEFAULT: "-fgnuc-version=4.2.1"

// RUN: %clang -c %s --target=i686-linux -fgnuc-version=100.99.99 -### 2>&1 | FileCheck %s --check-prefix=GNUC-OVERRIDE
// GNUC-OVERRIDE: "-fgnuc-version=100.99.99"

// RUN: %clang -c %s --target=i686-linux -fgnuc-version=0 -### 2>&1 | FileCheck %s --check-prefix=GNUC-DISABLE
// RUN: %clang -c %s --target=i686-linux -fgnuc-version= -### 2>&1 | FileCheck %s --check-prefix=GNUC-DISABLE
// GNUC-DISABLE-NOT: "-fgnuc-version=

// RUN: not %clang -c %s --target=i686-linux -fgnuc-version=100.100.10 2>&1 | FileCheck %s --check-prefix=GNUC-INVALID
// RUN: not %clang -c %s --target=i686-linux -fgnuc-version=100.10.100 2>&1 | FileCheck %s --check-prefix=GNUC-INVALID
// RUN: not %clang -c %s --target=i686-linux -fgnuc-version=-1.0.0 2>&1 | FileCheck %s --check-prefix=GNUC-INVALID
// GNUC-INVALID: error: invalid value {{.*}} in '-fgnuc-version={{.*}}'

// RUN: %clang -fgnuc-version=100.99.99 %s -dM -E -o - | FileCheck %s --check-prefix=GNUC-LARGE
// GNUC-LARGE: #define __GNUC_MINOR__ 99
// GNUC-LARGE: #define __GNUC_PATCHLEVEL__ 99
// GNUC-LARGE: #define __GNUC__ 100

// RUN: %clang -fgnuc-version=100.99.99 -x c++ %s -dM -E -o - | FileCheck %s --check-prefix=GXX-LARGE
// GXX-LARGE: #define __GNUG__ 100
