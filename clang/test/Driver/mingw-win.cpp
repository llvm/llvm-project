// REQUIRES: system-windows
// RUN: %clang -target i686-pc-windows-gnu -c -### --sysroot=c:\dummy %s 2>&1 | FileCheck -check-prefix=CHECK %s

// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++
// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++\\i686-w64-mingw32\\
// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++\\backward
// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++\\dummy\\
// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++\\dummy\\i686-w64-mingw32\\
// CHECK: c:\\dummy\\i686-w64-mingw32\\include\\c++\\dummy\\backward
// CHECK: c:\\dummy\\include\\c++\\dummy\\
// CHECK: c:\\dummy\\include\\c++\\dummy\\i686-w64-mingw32\\
// CHECK: c:\\dummy\\include\\c++\\dummy\\backward
// CHECK: c:\\dummy\\include\\c++\\
// CHECK: c:\\dummy\\include\\c++\\i686-w64-mingw32\\
// CHECK: c:\\dummy\\include\\c++\\backward
// CHECK: c:\\dummy\\include
// CHECK: c:\\dummy\\include-fixed
// CHECK: c:\\dummy\\i686-w64-mingw32\\include
// CHECK: c:\\dummy\\include
