// XFAIL: system-windows
// RUN: %clang -target x86_64-pc-windows-gnu -c -### --sysroot=/dummy/x86_64-w64-mingw32/5.1.0 %s 2>&1 | FileCheck -check-prefix=CHECK %s

// CHECK: /usr/x86_64-w64-mingw32/include/c++/
// CHECK: /usr/x86_64-w64-mingw32/include/c++/x86_64-w64-mingw32/
// CHECK: /usr/x86_64-w64-mingw32/include/c++/backward
// CHECK: /usr/x86_64-w64-mingw32/include/c++/5.1.0/
// CHECK: /usr/x86_64-w64-mingw32/include/c++/5.1.0/x86_64-w64-mingw32/
// CHECK: /usr/x86_64-w64-mingw32/include/c++/5.1.0/backward
// CHECK: /usr/include/c++/5.1.0/
// CHECK: /usr/include/c++/5.1.0/x86_64-w64-mingw32/
// CHECK: /usr/include/c++/5.1.0/backward
// CHECK: /dummy/x86_64-w64-mingw32/5.1.0/include/c++/
// CHECK: /dummy/x86_64-w64-mingw32/5.1.0/include/c++/x86_64-w64-mingw32/
// CHECK: /dummy/x86_64-w64-mingw32/5.1.0/include/c++/backward
// CHECK: /dummy/x86_64-w64-mingw32/5.1.0/include
// CHECK: /usr/x86_64-w64-mingw32/sys-root/mingw/include
// CHECK: /dummy/x86_64-w64-mingw32/5.1.0/include-fixed
// CHECK: /usr/x86_64-w64-mingw32/include
// CHECK: /usr/include
