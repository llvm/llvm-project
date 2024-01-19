// RUN: %clang -E -dM -x c++ %s | FileCheck -check-prefix=UNSPECIFIED %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=none -stdlib=libc++ %s | FileCheck -check-prefix=NONE %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=fast -stdlib=libc++ %s | FileCheck -check-prefix=FAST %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=extensive -stdlib=libc++ %s | FileCheck -check-prefix=EXTENSIVE %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=debug -stdlib=libc++ %s | FileCheck -check-prefix=DEBUG %s
// RUN: not %clang -x c++ -fstdlib-hardening=debug -stdlib=libstdc++ %s 2>&1 | FileCheck -check-prefix=ERROR %s
// RUN: not %clang -x c -fstdlib-hardening=debug %s 2>&1 | FileCheck -check-prefix=ERROR %s

// UNSPECIFIED-NOT: _STDLIB_HARDENING_MODE

// NONE: #define _STDLIB_HARDENING_MODE _STDLIB_HARDENING_MODE_NONE
// FAST: #define _STDLIB_HARDENING_MODE _STDLIB_HARDENING_MODE_FAST
// EXTENSIVE: #define _STDLIB_HARDENING_MODE _STDLIB_HARDENING_MODE_EXTENSIVE
// DEBUG: #define _STDLIB_HARDENING_MODE _STDLIB_HARDENING_MODE_DEBUG

// ERROR: standard library hardening is available only when libc++ is used
