// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix, system-windows
//
// clang-repl compiles position-independent code (it injects -fPIC), so a PCH
// built with a different PIC level is incompatible and must be rejected instead
// of silently accepted as a "compatible" language-option difference.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -fno-pic -Xclang -fincremental-extensions -target %host-jit-triple \
// RUN:     -Xclang -emit-pch -x c++-header -o %t/include.pch %t/include.hpp
//
// RUN: cat %t/main.cpp \
// RUN:     | not clang-repl -Xcc -include-pch -Xcc %t/include.pch 2>&1 \
// RUN:     | FileCheck %s

//--- include.hpp

int f_pch() { return 5; }

//--- main.cpp

extern "C" int printf(const char *, ...);
printf("f_pch = %d\n", f_pch());

// CHECK: incompatible with clang-repl's PIC level
