// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module A { header "A.h" export * }' > %t/module.modulemap
// RUN: echo '#define MY_MACRO 42' > %t/A.h
// RUN: echo '#include "A.h"' > %t/A.cpp

// Compile the module with embedded source files in C++ mode
// RUN: %clang_cc1 -fmodules -I%t -fmodules-cache-path=%t -fmodules-embed-all-files %t/module.modulemap -fmodule-name=A -x c++ -emit-module -o %t/A.pcm

// Remove the physical transient header file completely
// RUN: rm %t/A.h

// Verify that compilation compiles successfully without "file not found" errors in C++ mode
// by dynamically restoring the transient header into VFS from the embedded PCM.
// RUN: %clang_cc1 -fmodules -I%t -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap -fmodule-file=%t/A.pcm %t/A.cpp -emit-obj -o %t/A.o

// Test home-is-cwd mode with embedded files
// RUN: mkdir %t/cwd_test
// RUN: echo 'module B { header "B.h" export * }' > %t/cwd_test/module.modulemap
// RUN: echo '#define MY_MACRO_B 42' > %t/cwd_test/B.h
// RUN: echo '#include "B.h"' > %t/cwd_test/B.cpp
// RUN: cd %t/cwd_test && %clang_cc1 -fmodules -I. -fmodules-cache-path=. -fmodules-embed-all-files -fmodule-map-file-home-is-cwd module.modulemap -fmodule-name=B -x c++ -emit-module -o B.pcm
// RUN: rm %t/cwd_test/B.h
// RUN: cd %t/cwd_test && %clang_cc1 -fmodules -I. -fmodules-cache-path=. -fmodule-map-file-home-is-cwd -fmodule-map-file=module.modulemap -fmodule-file=B.pcm B.cpp -emit-obj -o B.o
