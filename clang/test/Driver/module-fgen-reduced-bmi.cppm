// It is annoying to handle different slash direction
// in Windows and Linux. So we disable the test on Windows
// here.
// REQUIRES: !system-windows
// On AIX, the default output for `-c` may be `.s` instead of `.o`,
// which makes the test fail. So disable the test on AIX.
// UNSUPPORTED: system-aix
//
// RUN: rm -rf %t && split-file %s %t && cd %t
//
// RUN: %clang -std=c++20 Hello.cppm -fmodule-output=Hello.pcm \
// RUN:     -fexperimental-modules-reduced-bmi -c -o Hello.o -### 2>&1 | FileCheck Hello.cppm
//
// RUN: %clang -std=c++20 Hello.cppm \
// RUN:     -fexperimental-modules-reduced-bmi -c -o Hello.o -### 2>&1 | \
// RUN:         FileCheck Hello.cppm --check-prefix=CHECK-UNSPECIFIED
//
// RUN: %clang -std=c++20 Hello.cppm \
// RUN:     -fexperimental-modules-reduced-bmi -c -### 2>&1 | \
// RUN:         FileCheck Hello.cppm --check-prefix=CHECK-NO-O
//
// RUN: %clang -std=c++20 Hello.cppm \
// RUN:     -fexperimental-modules-reduced-bmi -c -o AnotherName.o -### 2>&1 | \
// RUN:         FileCheck Hello.cppm --check-prefix=CHECK-ANOTHER-NAME
//
// RUN: %clang -std=c++20 Hello.cppm --precompile -fexperimental-modules-reduced-bmi \
// RUN:     -o Hello.full.pcm -### 2>&1 | FileCheck Hello.cppm \
// RUN:     --check-prefix=CHECK-EMIT-MODULE-INTERFACE
//
// RUN: %clang -std=c++20 Hello.cc -fexperimental-modules-reduced-bmi -Wall -Werror \
// RUN:     -c -o Hello.o -### 2>&1 | FileCheck Hello.cc

//--- Hello.cppm
export module Hello;

// Test that we won't generate the emit-module-interface as 2 phase compilation model.
// CHECK-NOT: -emit-module-interface
// CHECK: "-fexperimental-modules-reduced-bmi"

// CHECK-UNSPECIFIED: -fmodule-output=Hello.pcm

// CHECK-NO-O: -fmodule-output=Hello.pcm
// CHECK-ANOTHER-NAME: -fmodule-output=AnotherName.pcm

// With `-emit-module-interface` specified, we should still see the `-emit-module-interface`
// flag.
// CHECK-EMIT-MODULE-INTERFACE: -emit-module-interface

//--- Hello.cc

// CHECK-NOT: "-fexperimental-modules-reduced-bmi"
