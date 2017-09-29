// RUN: rm -rf %t
// RUN: %clang -cc1 -fsyntax-only -nobuiltininc -nostdinc++ -isysroot %S/Inputs/libc-libcxx/sysroot -isystem %S/Inputs/libc-libcxx/sysroot/usr/include/c++/v1 -isystem %S/Inputs/libc-libcxx/sysroot/usr/include -F%S/Inputs/libc-libcxx/sysroot/Frameworks -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c++ %s

#include <A/A.h>

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "module NonExistent1 { umbrella \"NonExistent\" }" > %t/modules.modulemap
// RUN: echo "" > %t/A.h
// RUN: echo "#include \"A.h\"  int i;" > %t/T.cxx
// RUN: %clang -I %t -fmodules -fsyntax-only %t/T.cxx
// expected-warning {{ umbrella directory }}
