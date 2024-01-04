// It is annoying to handle different slash direction
// in Windows and Linux. So we disable the test on Windows
// here.
// REQUIRES: !system-windows
// On AIX, the default output for `-c` may be `.s` instead of `.o`,
// which makes the test fail. So disable the test on AIX.
// REQUIRES: !system-aix
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 %t/Hello.cppm -fthinBMI-output=%t/Hello.pcm -c -o %t/Hello.o \
// RUN:   -### 2>&1 | FileCheck %t/Hello.cppm
// RUN: %clang -std=c++20 %t/Hello.cppm -fthinBMI-output=%t/Hello.pcm --precompile \
// RUN:   -o %t/Hello.pcm -### 2>&1 | FileCheck %t/Hello.cppm
//
// Tests that we can't use `-fthinBMI-output=` with multiple input files
// RUN: not %clang -std=c++20 %t/Hello.cppm %t/a.cppm -fthinBMI-output=%t/Hello.pcm  \
// RUN:     -o %t/a.out -### 2>&1 | FileCheck %t/a.cppm

//--- Hello.cppm
export module Hello;

// CHECK: "-emit-module-interface"{{.*}}"-fthinBMI-output={{[a-zA-Z0-9./-]*}}Hello.pcm"

//--- a.cppm
export module a;
// CHECK: cannot specify -fthinBMI-output when generating multiple module files
