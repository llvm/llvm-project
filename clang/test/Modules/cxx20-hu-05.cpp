// Test macro preservation in C++20 Header Units.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// Produce a pre-processed file.
// RUN: %clang_cc1 -std=c++20 -E -xc++-user-header hu-01.h -o hu-01.iih

// consume that to produce the heder unit.
// RUN: %clang_cc1 -std=c++20 -emit-header-unit \
// RUN: -xc++-header-unit-header-cpp-output hu-01.iih -o hu-01.pcm

// check that the header unit is named for the original file, not the .iih.
// RUN: %clang_cc1 -std=c++20 -module-file-info hu-01.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU %s -DTDIR=%t

//--- hu-01.h
#ifndef __GUARD
#define __GUARD

int baz(int);
#define FORTYTWO 42

#define SHOULD_NOT_BE_DEFINED -1
#undef SHOULD_NOT_BE_DEFINED

#endif // __GUARD

// CHECK-HU:  ====== C++20 Module structure ======
// CHECK-HU-NEXT:  Header Unit '.{{/|\\\\?}}hu-01.h' is the Primary Module at index #1
