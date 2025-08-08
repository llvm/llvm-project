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
// Tests that the .pcm file will be generated in the same directory with the specified
// output and the name of the .pcm file should be the same with the input file.
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output -c -o %t/output/Hello.o \
// RUN:   -fno-modules-reduced-bmi -### 2>&1 | FileCheck %t/Hello.cppm
//
// Tests that the output file will be generated in the input directory if the output
// file is not the corresponding object file.
// RUN: %clang -std=c++20 %t/Hello.cppm %t/AnotherModule.cppm -fmodule-output -o \
// RUN:   %t/output/a.out -fno-modules-reduced-bmi -### 2>&1 | FileCheck  %t/AnotherModule.cppm
//
// Tests that clang will reject the command line if it specifies -fmodule-output with
// multiple archs.
// RUN: not %clang %t/Hello.cppm -fmodule-output -arch i386 -arch x86_64 -### \
// RUN:   -fno-modules-reduced-bmi --target=x86_64-apple-darwin 2>&1 | FileCheck %t/Hello.cppm \
// RUN:   -check-prefix=MULTIPLE-ARCH

// Tests that the .pcm file will be generated in the same path with the specified one
// in the comamnd line.
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output=%t/pcm/Hello.pcm -o %t/Hello.o \
// RUN:   -fno-modules-reduced-bmi  -c -### 2>&1 | FileCheck %t/Hello.cppm --check-prefix=CHECK-SPECIFIED
//
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output=%t/Hello.pcm -fmodule-output -c -fsyntax-only \
// RUN:   -fno-modules-reduced-bmi  -### 2>&1 | FileCheck %t/Hello.cppm --check-prefix=CHECK-NOT-USED

// Test that we can emit a warning if the type of the input file is not a module interface unit.
// RUN: %clang -std=c++20 %t/a.cpp -fmodule-output -fno-modules-reduced-bmi  -c -o %t/a.o -### 2>&1 | FileCheck %t/a.cpp

//--- Hello.cppm
export module Hello;

// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/output/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/output/Hello.o" "-x" "pcm" "{{.*}}/output/Hello.pcm"

// MULTIPLE-ARCH: option '-fmodule-output' cannot be used with multiple arch options

// CHECK-SPECIFIED: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/pcm/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK-SPECIFIED: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello.o" "-x" "pcm" "{{.*}}/pcm/Hello.pcm"

// CHECK-NOT-USED-NOT: warning: argument unused during compilation: '-fmodule-output'
// CHECK-NOT-USED-NOT: warning: argument unused during compilation: '-fmodule-output=Hello.pcm'
// CHECK-NOT-USED-NOT: "-fmodule-output"
// CHECK-NOT-USED-NOT: "-fmodule-output="

//--- AnotherModule.cppm
export module AnotherModule;
// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello-{{.*}}.o" "-x" "pcm" "{{.*}}/Hello.pcm"
// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "AnotherModule.cppm" {{.*}}"-o" "{{.*}}/AnotherModule.pcm" "-x" "c++" "{{.*}}/AnotherModule.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "AnotherModule.cppm" {{.*}}"-o" "{{.*}}/AnotherModule-{{.*}}.o" "-x" "pcm" "{{.*}}/AnotherModule.pcm"

//--- a.cpp
export module a;

// CHECK: warning: argument unused during compilation: '-fmodule-output'
