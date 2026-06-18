// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -fmax-type-align=16 -Xclang -fdeprecated-macro -fno-stack-protector -Xclang -fwrapv -Xclang -fblocks -Xclang -fskip-odr-check-in-gmf -fexceptions -fcxx-exceptions -fgnuc-version=0 -target %host-jit-triple -Xclang -fblocks -Xclang -fmax-type-align=8 -Xclang -fincremental-extensions -Xclang -emit-pch -x c++-header -o %t/include.pch %t/include.hpp
//
// RUN: cat %t/main.cpp \
// RUN:     | clang-repl -Xcc -fgnuc-version=0 -Xcc -fno-stack-protector -Xcc -fwrapv -Xcc -fblocks -Xcc -fskip-odr-check-in-gmf -Xcc -fmax-type-align=8 -Xcc -include-pch -Xcc %t/include.pch \
// RUN:     | FileCheck %s

//--- include.hpp

int f_pch() { return 5; }

//--- main.cpp

extern "C" int printf(const char *, ...);
printf("f_pch = %d\n", f_pch());

// CHECK: f_pch = 5
