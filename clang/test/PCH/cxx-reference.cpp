// Test this without pch.
// RUN: %clang_cc1 -x c++ -triple %itanium_abi_triple -std=c++11 -include %S/cxx-reference.h -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -triple %itanium_abi_triple -std=c++11 -emit-pch -o %t %S/cxx-reference.h
// RUN: %clang_cc1 -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -emit-llvm -o - %s 
