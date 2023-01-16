// It is annoying to handle different slash direction
// in Windows and Linux. So we disable the test on Windows
// here.
// REQUIRES: !system-windows
// REQUIRES: x86-registered-target
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// Tests that the .pcm file will be generated in the same directory with the specified
// output and the name of the .pcm file should be the same with the input file.
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output -c -o %t/output/Hello.o \
// RUN:   -### 2>&1 | FileCheck %t/Hello.cppm
//
// Tests that the output file will be generated in the input directory if the output
// file is not the corresponding object file.
// RUN: %clang -std=c++20 %t/Hello.cppm %t/AnotherModule.cppm -fmodule-output -o \
// RUN:   %t/output/a.out -### 2>&1 | FileCheck  %t/AnotherModule.cppm
//
// Tests that clang will reject the command line if it specifies -fmodule-output with
// multiple archs.
// RUN: %clang %t/Hello.cppm -fmodule-output -arch i386 -arch x86_64 -### -target \
// RUN:   x86_64-apple-darwin 2>&1 | FileCheck %t/Hello.cppm -check-prefix=MULTIPLE-ARCH

// Tests that the .pcm file will be generated in the same path with the specified one
// in the comamnd line.
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output=%t/pcm/Hello.pcm -o %t/Hello.o \
// RUN:   -c -### 2>&1 | FileCheck %t/Hello.cppm --check-prefix=CHECK-SPECIFIED

//--- Hello.cppm
export module Hello;

// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/output/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/output/Hello.o" "-x" "pcm" "{{.*}}/output/Hello.pcm"

// MULTIPLE-ARCH: option '-fmodule-output' can't be used with multiple arch options

// CHECK-SPECIFIED: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/pcm/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK-SPECIFIED: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello.o" "-x" "pcm" "{{.*}}/pcm/Hello.pcm"

//--- AnotherModule.cppm
export module AnotherModule;
// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello.pcm" "-x" "c++" "{{.*}}/Hello.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "Hello.cppm" {{.*}}"-o" "{{.*}}/Hello-{{.*}}.o" "-x" "pcm" "{{.*}}/Hello.pcm"
// CHECK: "-emit-module-interface" {{.*}}"-main-file-name" "AnotherModule.cppm" {{.*}}"-o" "{{.*}}/AnotherModule.pcm" "-x" "c++" "{{.*}}/AnotherModule.cppm"
// CHECK: "-emit-obj" {{.*}}"-main-file-name" "AnotherModule.cppm" {{.*}}"-o" "{{.*}}/AnotherModule-{{.*}}.o" "-x" "pcm" "{{.*}}/AnotherModule.pcm"
