// RUN: %clang -E -dM -x c++ %s | FileCheck -check-prefix=UNSPECIFIED %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=none -stdlib=libc++ %s | FileCheck -check-prefix=NONE %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=fast -stdlib=libc++ %s | FileCheck -check-prefix=FAST %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=extensive -stdlib=libc++ %s | FileCheck -check-prefix=EXTENSIVE %s
// RUN: %clang -E -dM -x c++ -fstdlib-hardening=debug -stdlib=libc++ %s | FileCheck -check-prefix=DEBUG %s
// RUN: not %clang -x c++ -fstdlib-hardening=debug -stdlib=libstdc++ %s 2>&1 | FileCheck -check-prefix=ERROR %s
// RUN: not %clang -x c -fstdlib-hardening=debug %s 2>&1 | FileCheck -check-prefix=ERROR %s

// UNSPECIFIED-NOT: _LIBCPP_HARDENING_MODE

// NONE: #define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_NONE
// FAST: #define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_FAST
// EXTENSIVE: #define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_EXTENSIVE
// DEBUG: #define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_DEBUG

// ERROR: libc++ hardening is available only when libc++ is used
