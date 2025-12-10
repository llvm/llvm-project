// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header \
// RUN:  %t/a.h -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header \
// RUN:  %t/b.h -o %t/b.pcm -fmodule-file=%t/a.pcm
// RUN: echo "#define A2 44" >> %t/a.h
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header \
// RUN:  %t/a.h -o %t/a.v1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header \
// RUN:  %t/b.h -o %t/b.v1.pcm -fmodule-file=%t/a.v1.pcm
// RUN: not diff %t/b.pcm %t/b.v1.pcm &> /dev/null

//--- a.h
#pragma once
#define A 43

//--- b.h
#pragma once
import "a.h";
#define B 43
const int a = A;
