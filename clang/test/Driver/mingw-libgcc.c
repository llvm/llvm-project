// Test if mingw toolchain driver emits static linking (-lgcc -lgcc_eh) or dynamic linking (-lgcc_s -lgcc).
// Verified with gcc version 5.1.0 (i686-posix-dwarf-rev0, Built by MinGW-W64 project).

// gcc, static
// RUN: %clang -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefixes=CHECK_STATIC,CHECK_BDYNAMIC %s
// RUN: %clang -static -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefixes=CHECK_STATIC,CHECK_BSTATIC %s
// RUN: %clang -static-libgcc -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefixes=CHECK_STATIC,CHECK_BDYNAMIC %s
// RUN: %clang -static -shared -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefixes=CHECK_STATIC,CHECK_SHARED,CHECK_BSTATIC %s
// RUN: %clang -static-libgcc -shared -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefixes=CHECK_STATIC,CHECK_SHARED,CHECK_BDYNAMIC %s

// gcc, dynamic
// RUN: %clang -shared -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_DYNAMIC %s

// g++, static
// RUN: %clang -static --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_STATIC %s
// RUN: %clang -static-libgcc --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_STATIC %s
// RUN: %clang -static -shared --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_STATIC %s
// RUN: %clang -static-libgcc -shared --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_STATIC %s

// g++, dynamic
// RUN: %clang --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_DYNAMIC %s
// RUN: %clang -shared --driver-mode=g++ -v -target i686-pc-windows-gnu -rtlib=platform -### %s 2>&1 | FileCheck -check-prefix=CHECK_DYNAMIC %s

// CHECK_SHARED: "--shared"
// CHECK_BSTATIC: "-Bstatic"
// CHECK_BDYNAMIC: "-Bdynamic"
// CHECK_STATIC: "-lgcc" "-lgcc_eh"
// CHECK_DYNAMIC: "-lgcc_s" "-lgcc"
