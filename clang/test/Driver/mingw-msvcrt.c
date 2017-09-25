// RUN: %clang -v -target i686-pc-windows-gnu -### %s 2>&1 | FileCheck -check-prefix=CHECK_DEFAULT %s
// RUN: %clang -v -target i686-pc-windows-gnu -lmsvcr120 -### %s 2>&1 | FileCheck -check-prefix=CHECK_MSVCR120 %s

// CHECK_DEFAULT: "-lmingwex" "-lmsvcrt" "-ladvapi32"
// CHECK_MSVCR120: "-lmingwex" "-ladvapi32"
