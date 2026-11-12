// Ensure that the size of the reduced BMI is not larger than the full BMI
// in the most simple case. 

// This test requires linux commands.
// REQUIRES: system-linux

// RUN: rm -fr %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %s -o %t/a.reduced.pcm
//
// RUN: %python %S/compare-file-size.py %t/a.pcm %t/a.reduced.pcm

export module a;
