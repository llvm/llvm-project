// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.
//
// Note: see also cl-zc.cpp

// RUN: %clang_cl /TC /dev/null /E -Xclang -dM /Zc:__STDC__- 2>&1 | FileCheck %s --check-prefix=ZCSTDCIGNORED
// ZCSTDCIGNORED-NOT: #define __STDC__ 1
// ZCSTDCIGNORED: argument unused during compilation

// RUN: not %clang -Xclang -fno-ms-define-stdc %s 2>&1 | FileCheck %s --check-prefix="NOARG"
// NOARG: error: unknown argument: '-fno-ms-define-stdc'
