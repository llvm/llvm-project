// RUN: %clang -v -target i686-pc-windows-gnu -### %s 2>&1 | FileCheck -check-prefix=CHECK_DEFAULT %s
// RUN: %clang -v -target i686-pc-windows-gnu -### %s -lwindowsapp 2>&1 | FileCheck -check-prefix=CHECK_WINDOWSAPP %s

// CHECK_DEFAULT: "-lmsvcrt" "-ladvapi32" "-lshell32" "-luser32" "-lkernel32" "-lmingw32"
// CHECK_WINDOWSAPP: "-lwindowsapp" "-lmingw32"
// CHECK_WINDOWSAPP-SAME: "-lmsvcrt" "-lmingw32"
