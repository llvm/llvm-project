// REQUIRES: arm-registered-target
// UNSUPPORTED: system-windows
// UNSUPPORTED: llvm-driver

// RUN: mkdir -p %t

// RUN: ln -fs %clang %t/clang++
// RUN: ln -fs %clang %t/clang++3.5.0
// RUN: ln -fs %clang %t/clang++-3.5
// RUN: ln -fs %clang %t/clang++-tot
// RUN: ln -fs %clang %t/clang-c++
// RUN: ln -fs %clang %t/clang-g++
// RUN: ln -fs %clang %t/c++
// RUN: ln -fs %clang %t/foo-clang++
// RUN: ln -fs %clang %t/foo-clang++-3.5
// RUN: ln -fs %clang %t/foo-clang++3.5
// RUN: %t/clang++          -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/clang++3.5.0     -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/clang++-3.5      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/clang++-tot      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/clang-c++        -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/clang-g++        -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/c++              -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/foo-clang++      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/foo-clang++-3.5  -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %t/foo-clang++3.5   -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// CXXMODE: "-x" "c++"


// RUN: ln -fs %clang %t/clang-cl
// RUN: ln -fs %clang %t/cl
// RUN: ln -fs %clang %t/cl.exe
// RUN: ln -fs %clang %t/clang-cl3.5
// RUN: ln -fs %clang %t/clang-cl-3.5
// Note: use -- in front of the filename so it's not mistaken for an option on
// filesystems that use slashes for dir separators.
// RUN: %t/clang-cl         -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %t/cl               -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %t/cl.exe           -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %t/clang-cl3.5      -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %t/clang-cl-3.5     -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// CLMODE: "-fdiagnostics-format" "msvc"


// RUN: ln -fs %clang %t/clang-cpp
// RUN: ln -fs %clang %t/cpp
// RUN: %t/clang-cpp        -### %s 2>&1 | FileCheck -check-prefix=CPPMODE %s
// RUN: %t/cpp              -### %s 2>&1 | FileCheck -check-prefix=CPPMODE %s
// CPPMODE: "-E"


// RUN: ln -fs %clang %t/cl-clang
// RUN: %t/cl-clang        -### %s 2>&1 | FileCheck -check-prefix=CMODE %s
// CMODE: "-x" "c"
// CMODE-NOT: "-fdiagnostics-format" "msvc"


// RUN: ln -fs %clang %t/arm-linux-gnueabi-clang
// RUN: %t/arm-linux-gnueabi-clang -### %s 2>&1 | FileCheck -check-prefix=TARGET %s
// TARGET: Target: arm-unknown-linux-gnueabi
